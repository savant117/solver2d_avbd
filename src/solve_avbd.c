// SPDX-FileCopyrightText: 2025 Chris Giles
// SPDX-License-Identifier: MIT

#include "allocate.h"
#include "body.h"
#include "contact.h"
#include "core.h"
#include "joint.h"
#include "solvers.h"
#include "stack_allocator.h"
#include "world.h"

#include <stdbool.h>

#define PENALTY_MIN 1000.0f
#define PENALTY_MAX 10000000.0f
#define LAMBDA_MAX 10000000.0f
#define BETA 1000000.0f
#define ALPHA 0.95f
#define GAMMA 0.99f

static s2Vec3 solve_LDLT(s2Mat33 a, s2Vec3 b)
{
	// Compute LDL^T decomposition
	float D1 = a.cx.x;
	float L21 = a.cx.y / a.cx.x;
	float L31 = a.cx.z / a.cx.x;
	float D2 = a.cy.y - L21 * L21 * D1;
	float L32 = (a.cy.z - L21 * L31 * D1) / D2;
	float D3 = a.cz.z - (L31 * L31 * D1 + L32 * L32 * D2);

	// Forward substitution: Solve Ly = b
	float y1 = b.x;
	float y2 = b.y - L21 * y1;
	float y3 = b.z - L31 * y1 - L32 * y2;

	// Diagonal solve: Solve Dz = y
	float z1 = y1 / D1;
	float z2 = y2 / D2;
	float z3 = y3 / D3;

	// Backward substitution: Solve L^T x = z
	s2Vec3 x;
	x.z = z3;
	x.y = z2 - L32 * x.z;
	x.x = z1 - L21 * x.y - L31 * x.z;

	return x;
}

// Augmented Vertex Block Descent, 2025
// Chris Giles, Elie Diaz, Cem Yuksel
void s2Solve_AVBD(s2World* world, s2StepContext* context)
{
	s2Contact* contacts = world->contacts;
	int contactCapacity = world->contactPool.capacity;

	s2Joint* joints = world->joints;
	int jointCapacity = world->jointPool.capacity;

	s2Body* bodies = world->bodies;
	int bodyCapacity = world->bodyPool.capacity;

	float dt = context->dt;
	float inv_dt = 1.0f / dt;
	float inv_dt2 = inv_dt * inv_dt;

	// Compute constraint rows for contacts
	/*
	for (int i = 0; i < contactCapacity; ++i)
	{
		s2Contact* contact = contacts + i;
		if (s2IsFree(&contact->object) || contact->manifold.pointCount == 0)
		{
			continue;
		}
	}
	*/

	// Warmstart and initialize joints
	for (int i = 0; i < jointCapacity; ++i)
	{
		s2Joint* joint = joints + i;
		if (s2IsFree(&joint->object))
		{
			continue;
		}
		
		if (joint->type == s2_revoluteJoint)
		{
			s2Body* bodyA = context->bodies + joint->edges[0].bodyIndex;
			s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

			joint->revoluteJoint.localAnchorA = s2Sub(joint->localOriginAnchorA, bodyA->localCenter);
			joint->revoluteJoint.localAnchorB = s2Sub(joint->localOriginAnchorB, bodyB->localCenter);

			// Warmstart
			joint->revoluteJoint.impulse = s2MulSV(ALPHA * GAMMA, joint->revoluteJoint.impulse);
			joint->revoluteJoint.penalty = s2ClampSV(s2MulSV(GAMMA, joint->revoluteJoint.penalty), PENALTY_MIN, PENALTY_MAX);

			// Compute C(x-)
			s2Vec2 rAW = s2Add(s2RotateVector(bodyA->rot, joint->revoluteJoint.localAnchorA), bodyA->position);
			s2Vec2 rBW = s2Add(s2RotateVector(bodyB->rot, joint->revoluteJoint.localAnchorB), bodyB->position);

			joint->revoluteJoint.c0 = s2Sub(rBW, rAW);
		}
		else if (joint->type == s2_mouseJoint)
		{
			s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

			joint->mouseJoint.localAnchorB = s2Sub(joint->localOriginAnchorB, bodyB->localCenter);

			// Warmstart
			joint->mouseJoint.impulse = s2MulSV(joint->mouseJoint.biasCoefficient * GAMMA, joint->mouseJoint.impulse);
			joint->mouseJoint.penalty = s2ClampSV(s2MulSV(GAMMA, joint->mouseJoint.penalty), PENALTY_MIN, PENALTY_MAX);

			// Compute C(x-)
			s2Vec2 rAW = joint->mouseJoint.targetA;
			s2Vec2 rBW = s2Add(s2RotateVector(bodyB->rot, joint->localOriginAnchorB), bodyB->position);

			joint->mouseJoint.c0 = s2Sub(rBW, rAW);
		}
	}

	// Warmstart and initialize bodies
	for (int i = 0; i < bodyCapacity; ++i)
	{
		s2Body* body = bodies + i;
		if (s2IsFree(&body->object) || body->type == s2_staticBody)
		{
			continue;
		}

		body->inertialPosition = s2MulAdd(s2MulAdd(body->position, dt, body->linearVelocity), dt * dt, world->gravity);
		body->inertialRot = s2IntegrateRot(body->rot, dt * body->angularVelocity);

		body->deltaPosition0 = body->position;
		body->rot0 = body->rot;

		body->position = body->inertialPosition;
		body->rot = body->inertialRot;

		// TODO damping, adaptive warmstart, delta posiitons
	}

	for (int it = 0; it < context->iterations; ++it)
	{
		// Primal update
		for (int i = 0; i < bodyCapacity; ++i)
		{
			s2Body* body = bodies + i;
			if (s2IsFree(&body->object) || body->type == s2_staticBody)
			{
				continue;
			}

			// Initialize LHS and RHS
			s2Mat33 lhs = {0};
			lhs.cx.x = lhs.cy.y = body->mass * inv_dt * inv_dt;
			lhs.cz.z = body->I * inv_dt * inv_dt;

			s2Vec3 rhs = {body->mass * inv_dt2, body->mass * inv_dt2, body->I * inv_dt2};
			s2Vec2 dp = s2Sub(body->position, body->inertialPosition);
			rhs = s2Mul3(rhs, s2MakeVec3(dp.x, dp.y, s2ComputeAngularVelocity(body->rot, body->inertialRot, -1.0f)));

			// Accumulate forces and hessian
			for (int i = 0; i < jointCapacity; ++i)
			{
				s2Joint* joint = joints + i;
				if (s2IsFree(&joint->object))
				{
					continue;
				}

				s2Body* bodyA = context->bodies + joint->edges[0].bodyIndex;
				s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

				if (bodyA != body && bodyB != body)
				{
					continue;
				}

				if (joint->type == s2_revoluteJoint)
				{
					s2Vec2 rA = s2RotateVector(bodyA->rot, joint->revoluteJoint.localAnchorA);
					s2Vec2 rB = s2RotateVector(bodyB->rot, joint->revoluteJoint.localAnchorB);

					s2Vec2 rAW = s2Add(rA, bodyA->position);
					s2Vec2 rBW = s2Add(rB, bodyB->position);

					s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -ALPHA, joint->revoluteJoint.c0);

					s2Vec2 F = s2Add(joint->revoluteJoint.impulse, s2Mul(joint->revoluteJoint.penalty, C));

					s2Vec3 J1, J2;
					float H1, H2;
					if (body == bodyA)
					{
						J1 = s2MakeVec3(-1.0f, 0.0f, rA.y);
						J2 = s2MakeVec3(0.0f, -1.0f, -rA.x);
						H1 = rA.x;
						H2 = rA.y;
					}
					else
					{
						J1 = s2MakeVec3(1.0f, 0.0f, -rB.y);
						J2 = s2MakeVec3(0.0f, 1.0f, rB.x);
						H1 = -rB.x;
						H2 = -rB.y;
					}

					rhs = s2MulAdd3(rhs, F.x, J1);
					rhs = s2MulAdd3(rhs, F.y, J2);

					lhs = s2AddScaledOuter(lhs, joint->revoluteJoint.penalty.x, J1);
					lhs = s2AddScaledOuter(lhs, joint->revoluteJoint.penalty.y, J2);

					lhs.cz.z += abs(H1 * F.x + H2 * F.y);
				}
				else if (joint->type == s2_mouseJoint)
				{
					s2Vec2 rB = s2RotateVector(bodyB->rot, joint->mouseJoint.localAnchorB);

					s2Vec2 rAW = joint->mouseJoint.targetA;
					s2Vec2 rBW = s2Add(rB, bodyB->position);

					s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -joint->mouseJoint.biasCoefficient, joint->mouseJoint.c0);

					s2Vec2 F = s2Add(joint->mouseJoint.impulse, s2Mul(joint->mouseJoint.penalty, C));

					s2Vec3 J1, J2;
					float H1, H2;

					J1 = s2MakeVec3(1.0f, 0.0f, -rB.y);
					J2 = s2MakeVec3(0.0f, 1.0f, rB.x);
					H1 = -rB.x;
					H2 = -rB.y;

					rhs = s2MulAdd3(rhs, F.x, J1);
					rhs = s2MulAdd3(rhs, F.y, J2);

					lhs = s2AddScaledOuter(lhs, joint->mouseJoint.penalty.x, J1);
					lhs = s2AddScaledOuter(lhs, joint->mouseJoint.penalty.y, J2);

					lhs.cz.z += abs(H1 * F.x + H2 * F.y);
				}
			}

			// Solve and update position
			s2Vec3 dx = solve_LDLT(lhs, rhs);
			body->position = s2Sub(body->position, s2MakeVec2(dx.x, dx.y));
			body->rot = s2IntegrateRot(body->rot, -dx.z);
		}

		// Dual update
		for (int i = 0; i < jointCapacity; ++i)
		{
			s2Joint* joint = joints + i;
			if (s2IsFree(&joint->object))
			{
				continue;
			}

			if (joint->type == s2_revoluteJoint)
			{
				s2Body* bodyA = context->bodies + joint->edges[0].bodyIndex;
				s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

				s2Vec2 rA = s2RotateVector(bodyA->rot, joint->revoluteJoint.localAnchorA);
				s2Vec2 rB = s2RotateVector(bodyB->rot, joint->revoluteJoint.localAnchorB);

				s2Vec2 rAW = s2Add(rA, bodyA->position);
				s2Vec2 rBW = s2Add(rB, bodyB->position);

				s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -ALPHA, joint->revoluteJoint.c0);

				joint->revoluteJoint.impulse = s2MinSV(s2Add(joint->revoluteJoint.impulse, s2Mul(joint->revoluteJoint.penalty, C)), LAMBDA_MAX);
				joint->revoluteJoint.penalty = s2MinSV(s2MulAdd(joint->revoluteJoint.penalty, BETA, s2Abs(C)), PENALTY_MAX);
			}
			else if (joint->type == s2_mouseJoint)
			{
				s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

				s2Vec2 rB = s2RotateVector(bodyB->rot, joint->mouseJoint.localAnchorB);

				s2Vec2 rAW = joint->mouseJoint.targetA;
				s2Vec2 rBW = s2Add(rB, bodyB->position);

				s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -joint->mouseJoint.biasCoefficient, joint->mouseJoint.c0);

				joint->mouseJoint.impulse = s2MinSV(s2Add(joint->mouseJoint.impulse, s2Mul(joint->mouseJoint.penalty, C)), LAMBDA_MAX);
				joint->mouseJoint.penalty = s2MinSV(s2MulAdd(joint->mouseJoint.penalty, BETA, s2Abs(C)), PENALTY_MAX);
			}
		}
	}

	// Integration
	for (int i = 0; i < bodyCapacity; ++i)
	{
		s2Body* body = bodies + i;
		if (s2IsFree(&body->object))
		{
			continue;
		}

		if (body->type != s2_dynamicBody)
		{
			continue;
		}

		body->prevLinearVelocity = body->linearVelocity;
		body->linearVelocity = s2MulSV(inv_dt, s2Sub(body->position, body->deltaPosition0));
		body->angularVelocity = s2ComputeAngularVelocity(body->rot, body->rot0, -inv_dt);
	}
}
