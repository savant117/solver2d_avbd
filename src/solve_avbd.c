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

#define PENALTY_MIN 10000.0f
#define PENALTY_MAX 10000000.0f
#define LAMBDA_MAX 10000000.0f
#define ALLOWED_PENETRATION s2_linearSlop
#define BETA 200000.0f
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

	// Warmstart and initialize contacts
	for (int i = 0; i < contactCapacity; ++i)
	{
		s2Contact* contact = contacts + i;
		if (s2IsFree(&contact->object) || contact->manifold.pointCount == 0)
		{
			continue;
		}

		s2Body* bodyA = context->bodies + contact->edges[0].bodyIndex;
		s2Body* bodyB = context->bodies + contact->edges[1].bodyIndex;

		for (int j = 0; j < contact->manifold.pointCount; j++)
		{
			s2ManifoldPoint* point = contact->manifold.points + j;
			if (point->separation > 0)
			{
				continue;
			}

			point->localAnchorA = s2Sub(point->localOriginAnchorA, bodyA->localCenter);
			point->localAnchorB = s2Sub(point->localOriginAnchorB, bodyB->localCenter);

			// Warmstart
			point->normalImpulse = ALPHA * GAMMA * point->normalImpulse;
			point->tangentImpulse = ALPHA * GAMMA * point->tangentImpulse;

			point->penalty.x = S2_CLAMP((GAMMA * point->penalty.x), PENALTY_MIN, PENALTY_MAX);
			point->penalty.y = S2_CLAMP((GAMMA * point->penalty.y), PENALTY_MIN, PENALTY_MAX);
		}
	}

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
			s2Vec2 rA = s2RotateVector(bodyA->rot, joint->revoluteJoint.localAnchorA);
			s2Vec2 rB = s2RotateVector(bodyB->rot, joint->revoluteJoint.localAnchorB);

			joint->revoluteJoint.c0 = s2Add(s2Sub(rB, rA), s2Sub(bodyB->position, bodyA->position));
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

		s2Vec2 gravity = s2MulSV(body->gravityScale, world->gravity);
		s2Vec2 force = s2MulAdd(gravity, body->invMass, body->force);
		body->deltaInertialPosition = s2MulAdd(s2MulSV(dt, body->linearVelocity), dt * dt, force);
		body->deltaInertialRot =  dt * body->angularVelocity + dt * dt * body->invI * body->torque;

		// Adaptive warmstart
		float accelWeight = 1.0f;
		if (s2LengthSquared(force) > 0)
		{
			s2Vec2 accel = s2MulSV(inv_dt, s2Sub(body->linearVelocity, body->prevLinearVelocity));
			float accelExt = s2Dot(accel, s2Normalize(force));
			float accelWeight = accelExt / s2Length(force);
			accelWeight = S2_CLAMP(accelWeight, 0.0f, 1.0f);
		}

		body->deltaPosition = s2MulAdd(s2MulSV(dt, body->linearVelocity), accelWeight * dt * dt, force);
		body->deltaRot = body->deltaInertialRot;
	}

	for (int it = 0; it < context->iterations; ++it)
	{
		// Primal update
		for (int bi = 0; bi < bodyCapacity; ++bi)
		{
			s2Body* body = bodies + bi;
			if (s2IsFree(&body->object) || body->type == s2_staticBody)
			{
				continue;
			}

			// Initialize LHS and RHS
			s2Mat33 lhs = {0};
			lhs.cx.x = lhs.cy.y = body->mass * inv_dt * inv_dt;
			lhs.cz.z = body->I * inv_dt * inv_dt;

			s2Vec3 rhs = {body->mass * inv_dt2, body->mass * inv_dt2, body->I * inv_dt2};
			s2Vec2 dp = s2Sub(body->deltaPosition, body->deltaInertialPosition);
			rhs = s2Mul3(rhs, s2MakeVec3(dp.x, dp.y, body->deltaRot - body->deltaInertialRot));

			// Accumulate forces and hessian for contacts
			// TODO iterate over only contacts connected to body
			for (int i = 0; i < contactCapacity; ++i)
			{
				s2Contact* contact = contacts + i;
				if (s2IsFree(&contact->object) || contact->manifold.pointCount == 0)
				{
					continue;
				}

				s2Body* bodyA = context->bodies + contact->edges[0].bodyIndex;
				s2Body* bodyB = context->bodies + contact->edges[1].bodyIndex;

				if (bodyA != body && bodyB != body)
				{
					continue;
				}

				for (int j = 0; j < contact->manifold.pointCount; j++)
				{
					s2ManifoldPoint* point = contact->manifold.points + j;
					if (point->separation > 0)
					{
						continue;
					}

					s2Vec2 rA = s2RotateVector(bodyA->rot, point->localAnchorA);
					s2Vec2 rB = s2RotateVector(bodyB->rot, point->localAnchorB);

					s2Vec2 N = contact->manifold.normal;
					s2Vec2 T = s2RightPerp(N);

					s2Vec3 J1A, J1B, J2A, J2B, J1, J2;
					J1A = s2MakeVec3(-N.x, -N.y, -s2Cross(rA, N));
					J1B = s2MakeVec3(N.x, N.y, s2Cross(rB, N));
					J2A = s2MakeVec3(-T.x, -T.y, -s2Cross(rA, T));
					J2B = s2MakeVec3(T.x, T.y, s2Cross(rB, T));

					if (body == bodyA)
					{
						J1 = J1A;
						J2 = J2A;
					}
					else
					{
						J1 = J1B;
						J2 = J2B;
					}

					s2Vec3 dpA = s2MakeVec3(bodyA->deltaPosition.x, bodyA->deltaPosition.y, bodyA->deltaRot);
					s2Vec3 dpB = s2MakeVec3(bodyB->deltaPosition.x, bodyB->deltaPosition.y, bodyB->deltaRot);

					s2Vec2 C;
					C.x =
						s2Dot3(J1A, dpA) + s2Dot3(J1B, dpB) + (point->separation + ALLOWED_PENETRATION) * (1 - ALPHA);
					C.y = s2Dot3(J2A, dpA) + s2Dot3(J2B, dpB);

					s2Vec2 F = s2Add(s2MakeVec2(point->normalImpulse, point->tangentImpulse), s2Mul(point->penalty, C));

					F.x = S2_MIN(F.x, 0.0f);
					float bounds = fabsf(F.x) * contact->friction;

					float kf = point->penalty.y;
					if (F.y != 0)
					{
						if (F.y >= bounds)
							kf = fabsf(bounds * point->penalty.y / F.y);
						else if (F.y <= -bounds)
							kf = fabsf(bounds * point->penalty.y / F.y);
					}

					F.y = S2_CLAMP(F.y, -bounds, bounds);

					rhs = s2MulAdd3(rhs, F.x, J1);
					rhs = s2MulAdd3(rhs, F.y, J2);

					lhs = s2AddScaledOuter(lhs, point->penalty.x, J1);
					lhs = s2AddScaledOuter(lhs, kf, J2);
				}
			}

			// Accumulate forces and hessian for joints
			// TODO iterate over only joints connected to body
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
					s2Vec2 rA = s2RotateVector(s2IntegrateRot(bodyA->rot, bodyA->deltaRot), joint->revoluteJoint.localAnchorA);
					s2Vec2 rB = s2RotateVector(s2IntegrateRot(bodyB->rot, bodyB->deltaRot), joint->revoluteJoint.localAnchorB);

					s2Vec2 rAW = s2Add(rA, bodyA->deltaPosition);
					s2Vec2 rBW = s2Add(rB, bodyB->deltaPosition);

					s2Vec2 C = s2MulAdd(s2Add(s2Sub(rBW, rAW), s2Sub(bodyB->position, bodyA->position)), -ALPHA, joint->revoluteJoint.c0);

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

					lhs.cz.z += fabsf(H1 * F.x + H2 * F.y);

					// TODO joint limits
				}
				else if (joint->type == s2_mouseJoint)
				{
					s2Vec2 rB = s2RotateVector(s2IntegrateRot(bodyB->rot, bodyB->deltaRot), joint->mouseJoint.localAnchorB);

					s2Vec2 rAW = joint->mouseJoint.targetA;
					s2Vec2 rBW = s2Add(rB, s2Add(bodyB->position, bodyB->deltaPosition));

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

					lhs.cz.z += fabsf(H1 * F.x + H2 * F.y);
				}
			}

			// Solve and update position
			s2Vec3 dx = solve_LDLT(lhs, rhs);
			body->deltaPosition = s2Sub(body->deltaPosition, s2MakeVec2(dx.x, dx.y));
			body->deltaRot -= dx.z;
		}

		// Dual update contacts
		for (int i = 0; i < contactCapacity; ++i)
		{
			s2Contact* contact = contacts + i;
			if (s2IsFree(&contact->object) || contact->manifold.pointCount == 0)
			{
				continue;
			}

			s2Body* bodyA = context->bodies + contact->edges[0].bodyIndex;
			s2Body* bodyB = context->bodies + contact->edges[1].bodyIndex;

			for (int j = 0; j < contact->manifold.pointCount; j++)
			{
				s2ManifoldPoint* point = contact->manifold.points + j;
				if (point->separation > 0)
				{
					continue;
				}

				s2Vec2 rA = s2RotateVector(bodyA->rot, point->localAnchorA);
				s2Vec2 rB = s2RotateVector(bodyB->rot, point->localAnchorB);

				s2Vec2 N = contact->manifold.normal;
				s2Vec2 T = s2RightPerp(N);

				s2Vec3 J1A, J1B, J2A, J2B;
				J1A = s2MakeVec3(-N.x, -N.y, -s2Cross(rA, N));
				J1B = s2MakeVec3(N.x, N.y, s2Cross(rB, N));
				J2A = s2MakeVec3(-T.x, -T.y, -s2Cross(rA, T));
				J2B = s2MakeVec3(T.x, T.y, s2Cross(rB, T));

				s2Vec3 dpA = s2MakeVec3(bodyA->deltaPosition.x, bodyA->deltaPosition.y, bodyA->deltaRot);
				s2Vec3 dpB = s2MakeVec3(bodyB->deltaPosition.x, bodyB->deltaPosition.y, bodyB->deltaRot);

				s2Vec2 C;
				C.x =
					s2Dot3(J1A, dpA) + s2Dot3(J1B, dpB) + (point->separation + ALLOWED_PENETRATION) * (1 - ALPHA);
				C.y = s2Dot3(J2A, dpA) + s2Dot3(J2B, dpB);

				s2Vec2 F = s2Add(s2MakeVec2(point->normalImpulse, point->tangentImpulse), s2Mul(point->penalty, C));

				F.x = S2_MIN(F.x, 0.0f);
				float bounds = fabsf(F.x) * contact->friction;
				F.y = S2_CLAMP(F.y, -bounds, bounds);

				point->normalImpulse = S2_CLAMP(F.x, -LAMBDA_MAX, LAMBDA_MAX);
				point->tangentImpulse = S2_CLAMP(F.y, -LAMBDA_MAX, LAMBDA_MAX);

				if (F.x < 0)
					point->penalty.x = S2_MIN((point->penalty.x + fabsf(C.x) * BETA), PENALTY_MAX);
				if (F.y > -bounds && F.y < bounds)
					point->penalty.y = S2_MIN((point->penalty.y + fabsf(C.y) * BETA), PENALTY_MAX);
			}
		}

		// Dual update joints
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

				s2Vec2 rA = s2RotateVector(s2IntegrateRot(bodyA->rot, bodyA->deltaRot), joint->revoluteJoint.localAnchorA);
				s2Vec2 rB = s2RotateVector(s2IntegrateRot(bodyB->rot, bodyB->deltaRot), joint->revoluteJoint.localAnchorB);

				s2Vec2 rAW = s2Add(rA, bodyA->deltaPosition);
				s2Vec2 rBW = s2Add(rB, bodyB->deltaPosition);

				s2Vec2 C = s2MulAdd(s2Add(s2Sub(rBW, rAW), s2Sub(bodyB->position, bodyA->position)), -ALPHA, joint->revoluteJoint.c0);

				joint->revoluteJoint.impulse = s2ClampSV(
					s2Add(joint->revoluteJoint.impulse, s2Mul(joint->revoluteJoint.penalty, C)), -LAMBDA_MAX, LAMBDA_MAX);

				joint->revoluteJoint.penalty = s2MinSV(s2MulAdd(joint->revoluteJoint.penalty, BETA, s2Abs(C)), PENALTY_MAX);
			}
			else if (joint->type == s2_mouseJoint)
			{
				s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

				s2Vec2 rB = s2RotateVector(s2IntegrateRot(bodyB->rot, bodyB->deltaRot), joint->mouseJoint.localAnchorB);

				s2Vec2 rAW = joint->mouseJoint.targetA;
				s2Vec2 rBW = s2Add(rB, s2Add(bodyB->position, bodyB->deltaPosition));

				s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -joint->mouseJoint.biasCoefficient, joint->mouseJoint.c0);

				joint->mouseJoint.impulse =
					s2ClampSV(s2Add(joint->mouseJoint.impulse, s2Mul(joint->mouseJoint.penalty, C)), -LAMBDA_MAX, LAMBDA_MAX);

				joint->mouseJoint.penalty = s2MinSV(s2MulAdd(joint->mouseJoint.penalty, BETA, s2Abs(C)), PENALTY_MAX);
			}
		}
	}

	// Compute velocity and update positions
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
		body->linearVelocity = s2MulSV(inv_dt, body->deltaPosition);
		body->angularVelocity = inv_dt * body->deltaRot;

		body->linearVelocity = s2MulSV(1.0f / (1.0f + dt * body->linearDamping), body->linearVelocity);
		body->angularVelocity *= 1.0f / (1.0f + dt * body->angularDamping);

		body->position = s2Add(body->position, body->deltaPosition);
		body->rot = s2IntegrateRot(body->rot, body->deltaRot);
	}
}
