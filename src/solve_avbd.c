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
#define ALLOWED_PENETRATION 0.00025f
#define BAUMGARTE

#ifdef BAUMGARTE
	// These only matter when using baumgarte
	#define BETA 1000000.0f
	#define ALPHA 0.99f
	#define GAMMA 0.99f
#else
	#define BETA 2.0f
	#define ALPHA 0.9f
#endif

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
			#ifdef BAUMGARTE
				point->normalImpulse = ALPHA * GAMMA * point->normalImpulse;
				point->tangentImpulse = ALPHA * GAMMA * point->tangentImpulse;
				point->penalty.x = S2_CLAMP((GAMMA * point->penalty.x), PENALTY_MIN, PENALTY_MAX);
				point->penalty.y = S2_CLAMP((GAMMA * point->penalty.y), PENALTY_MIN, PENALTY_MAX);
			#else
			point->penalty.x = S2_CLAMP(fabsf(point->normalImpulse) * BETA * inv_dt, PENALTY_MIN, PENALTY_MAX);
			point->penalty.y = S2_CLAMP(fabsf(point->tangentImpulse) * BETA * inv_dt, PENALTY_MIN, PENALTY_MAX);
			#endif

			// Compute C(x-)
			s2Vec2 rAW = s2Add(s2RotateVector(bodyA->rot, point->localAnchorA), bodyA->position);
			s2Vec2 rBW = s2Add(s2RotateVector(bodyB->rot, point->localAnchorB), bodyB->position);

			rAW = s2MulAdd(rAW, -point->separation, contact->manifold.normal);
			rBW = s2MulAdd(rBW, point->separation, contact->manifold.normal);

			s2Vec2 dp = s2Sub(rBW, rAW);
			point->c0.x = s2Dot(dp, contact->manifold.normal) - ALLOWED_PENETRATION;
			point->c0.y = s2Dot(dp, s2RightPerp(contact->manifold.normal));
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
			#ifdef BAUMGARTE
				joint->revoluteJoint.impulse = s2MulSV(ALPHA * GAMMA, joint->revoluteJoint.impulse);
				joint->revoluteJoint.penalty = s2ClampSV(s2MulSV(GAMMA, joint->revoluteJoint.penalty), PENALTY_MIN, PENALTY_MAX);
			#else
			joint->revoluteJoint.penalty =
				s2ClampSV(s2MulSV(BETA * inv_dt, s2Abs(joint->revoluteJoint.impulse)), PENALTY_MIN, PENALTY_MAX);
			#endif

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
			#ifdef BAUMGARTE
				joint->mouseJoint.impulse = s2MulSV(joint->mouseJoint.biasCoefficient * GAMMA, joint->mouseJoint.impulse);
				joint->mouseJoint.penalty = s2ClampSV(s2MulSV(GAMMA, joint->mouseJoint.penalty), PENALTY_MIN, PENALTY_MAX);
			#else
			joint->mouseJoint.penalty =
				s2ClampSV(s2MulSV(BETA * inv_dt, s2Abs(joint->mouseJoint.impulse)), PENALTY_MIN, PENALTY_MAX);
			#endif

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
		body->inertialPosition = s2MulAdd(s2MulAdd(body->position, dt, body->linearVelocity), dt * dt, force);
		body->inertialRot = s2IntegrateRot(body->rot, dt * (body->angularVelocity + body->invI * body->torque));

		body->deltaPosition0 = body->position;
		body->rot0 = body->rot;

		// Adaptive warmstart
		float accelWeight = 1.0f;
		if (s2LengthSquared(gravity) > 0)
		{
			s2Vec2 accel = s2MulSV(inv_dt, s2Sub(body->linearVelocity, body->prevLinearVelocity));
			float accelExt = s2Dot(accel, s2Normalize(gravity));
			float accelWeight = accelExt / s2Length(gravity);
			accelWeight = S2_CLAMP(accelWeight, 0.0f, 1.0f);
		}

		body->position = s2MulAdd(s2MulAdd(body->position, dt, body->linearVelocity), accelWeight * dt * dt, gravity);
		body->rot = body->inertialRot;
	}

	#ifdef BAUMGARTE
	int totalIterations = context->iterations;
	#else
	int totalIterations = context->iterations + context->extraIterations;
	#endif

	for (int it = 0; it < totalIterations; ++it)
	{
		#ifdef BAUMGARTE
			float alpha = ALPHA;
		#else
			float alpha = it < context->iterations ? 1.0f : ALPHA;
		#endif

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
			s2Vec2 dp = s2Sub(body->position, body->inertialPosition);
			rhs = s2Mul3(rhs, s2MakeVec3(dp.x, dp.y, s2ComputeAngularVelocity(body->rot, body->inertialRot, -1.0f)));

			// Accumulate forces and hessian for contacts
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

					s2Vec2 rAW = s2Add(rA, bodyA->position);
					s2Vec2 rBW = s2Add(rB, bodyB->position);

					rAW = s2MulAdd(rAW, -point->separation, contact->manifold.normal);
					rBW = s2MulAdd(rBW, point->separation, contact->manifold.normal);

					s2Vec2 N = contact->manifold.normal;
					s2Vec2 T = s2RightPerp(N);

					s2Vec2 dp = s2Sub(rBW, rAW);
					s2Vec2 C;
					C.x = s2Dot(dp, N);
					C.y = s2Dot(dp, T);
					C = s2MulAdd(C, -alpha, point->c0);

					s2Vec2 F = s2Add(s2MakeVec2(point->normalImpulse, point->tangentImpulse), s2Mul(point->penalty, C));

					F.x = S2_MIN(F.x, 0.0f);
					float bounds = fabsf(F.x) * contact->friction;
					F.y = S2_CLAMP(F.y, -bounds, bounds);

					s2Vec3 J1, J2;
					if (body == bodyA)
					{
						J1 = s2MakeVec3(-N.x, -N.y, -s2Cross(rA, N));
						J2 = s2MakeVec3(-T.x, -T.y, -s2Cross(rA, T));
					}
					else
					{
						J1 = s2MakeVec3(N.x, N.y, s2Cross(rB, N));
						J2 = s2MakeVec3(T.x, T.y, s2Cross(rB, T));
					}

					rhs = s2MulAdd3(rhs, F.x, J1);
					rhs = s2MulAdd3(rhs, F.y, J2);

					lhs = s2AddScaledOuter(lhs, point->penalty.x, J1);
					lhs = s2AddScaledOuter(lhs, point->penalty.y, J2);
				}
			}

			// Accumulate forces and hessian for joints
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

					s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -alpha, joint->revoluteJoint.c0);

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

					lhs.cz.z += fabsf(H1 * F.x + H2 * F.y);
				}
			}

			// Solve and update position
			s2Vec3 dx = solve_LDLT(lhs, rhs);
			body->position = s2Sub(body->position, s2MakeVec2(dx.x, dx.y));
			body->rot = s2IntegrateRot(body->rot, -dx.z);
		}

		// Dual update contacts on primary iterations
		if (it < context->iterations)
		{
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

					s2Vec2 rAW = s2Add(rA, bodyA->position);
					s2Vec2 rBW = s2Add(rB, bodyB->position);

					rAW = s2MulAdd(rAW, -point->separation, contact->manifold.normal);
					rBW = s2MulAdd(rBW, point->separation, contact->manifold.normal);

					s2Vec2 N = contact->manifold.normal;
					s2Vec2 T = s2RightPerp(N);

					s2Vec2 dp = s2Sub(rBW, rAW);
					s2Vec2 C;
					C.x = s2Dot(dp, N);
					C.y = s2Dot(dp, T);
					C = s2MulAdd(C, -alpha, point->c0);

					s2Vec2 F = s2Add(s2MakeVec2(point->normalImpulse, point->tangentImpulse), s2Mul(point->penalty, C));

					F.x = S2_MIN(F.x, 0.0f);
					float bounds = fabsf(F.x) * contact->friction;
					F.y = S2_CLAMP(F.y, -bounds, bounds);

					point->normalImpulse = S2_CLAMP(F.x, -LAMBDA_MAX, LAMBDA_MAX);
					point->tangentImpulse = S2_CLAMP(F.y, -LAMBDA_MAX, LAMBDA_MAX);

#ifdef BAUMGARTE
					if (F.x < 0)
						point->penalty.x = S2_MIN((point->penalty.x + fabsf(C.x) * BETA), PENALTY_MAX);
					if (F.y > -bounds && F.y < bounds)
						point->penalty.y = S2_MIN((point->penalty.y + fabsf(C.y) * BETA), PENALTY_MAX);
#endif
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

					s2Vec2 rA = s2RotateVector(bodyA->rot, joint->revoluteJoint.localAnchorA);
					s2Vec2 rB = s2RotateVector(bodyB->rot, joint->revoluteJoint.localAnchorB);

					s2Vec2 rAW = s2Add(rA, bodyA->position);
					s2Vec2 rBW = s2Add(rB, bodyB->position);

					s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -alpha, joint->revoluteJoint.c0);

					joint->revoluteJoint.impulse = s2ClampSV(
						s2Add(joint->revoluteJoint.impulse, s2Mul(joint->revoluteJoint.penalty, C)), -LAMBDA_MAX, LAMBDA_MAX);

#ifdef BAUMGARTE
					joint->revoluteJoint.penalty = s2MinSV(s2MulAdd(joint->revoluteJoint.penalty, BETA, s2Abs(C)), PENALTY_MAX);
#endif
				}
				else if (joint->type == s2_mouseJoint)
				{
					s2Body* bodyB = context->bodies + joint->edges[1].bodyIndex;

					s2Vec2 rB = s2RotateVector(bodyB->rot, joint->mouseJoint.localAnchorB);

					s2Vec2 rAW = joint->mouseJoint.targetA;
					s2Vec2 rBW = s2Add(rB, bodyB->position);

					s2Vec2 C = s2MulAdd(s2Sub(rBW, rAW), -joint->mouseJoint.biasCoefficient, joint->mouseJoint.c0);

					joint->mouseJoint.impulse =
						s2ClampSV(s2Add(joint->mouseJoint.impulse, s2Mul(joint->mouseJoint.penalty, C)), -LAMBDA_MAX, LAMBDA_MAX);

#ifdef BAUMGARTE
					joint->mouseJoint.penalty = s2MinSV(s2MulAdd(joint->mouseJoint.penalty, BETA, s2Abs(C)), PENALTY_MAX);
#endif
				}
			}
		}

		if (it == context->iterations - 1)
		{
			// Integration on last primary iteration
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

				body->linearVelocity = s2MulSV(1.0f / (1.0f + dt * body->linearDamping), body->linearVelocity);
				body->angularVelocity *= 1.0f / (1.0f + dt * body->angularDamping);
			}
		}
	}
}
