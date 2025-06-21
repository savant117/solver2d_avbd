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

#define PENALTY_MIN 1000
#define PENALTY_MAX 10000000
#define PENALTY_SCALE 100

// Data needed for each AVBD constraint row
typedef struct s2ConstraintRow
{
	float c;		// Constraint error
	s2Vec3 J[2];	// Jacobian
	float G[2];		// Geometric stiffness (angular only required here, but in general this would be 3x3 matrix)
	float k;		// Penalty parameter
	float lambda;	// Dual variable
} s2ConstraintRow;

// Augmented Vertex Block Descent, 2025
// Chris Giles, Elie Diaz, Cem Yuksel
void s2Solve_AVBD(s2World* world, s2StepContext* context)
{
	s2Contact* contacts = world->contacts;
	int contactCapacity = world->contactPool.capacity;

	s2Joint* joints = world->joints;
	int jointCapacity = world->jointPool.capacity;

	// Allocate enough rows for all the contact and joint constraint rows
	s2ConstraintRow* rows = s2AllocateStackItem(world->stackAllocator, (contactCapacity * 2 + jointCapacity * 2) * sizeof(s2ConstraintRow), "constraint");
	int rowCount = 0;

	// Compute constraint rows for contacts
	for (int i = 0; i < contactCapacity; ++i)
	{
		s2Contact* contact = contacts + i;
		if (s2IsFree(&contact->object) || contact->manifold.pointCount == 0)
		{
			continue;
		}

		
	}

	// Compute constraint rows for joints
	for (int i = 0; i < jointCapacity; ++i)
	{
		s2Joint* joint = joints + i;
		if (s2IsFree(&joint->object))
		{
			continue;
		}
		
		if (joint->type == s2_revoluteJoint)
		{
			rows[rowCount].k = PENALTY_MIN;
			rows[rowCount].lambda = 0.0f;
		}
	}

	float h = context->dt;
	float inv_h = 1.0f / h;
	s2Body* bodies = world->bodies;
	int bodyCapacity = world->bodyPool.capacity;
	s2Vec2 gravity = world->gravity;

	/*

	for (int substep = 0; substep < substepCount; ++substep)
	{
		// Integrate velocities and positions
		for (int i = 0; i < bodyCapacity; ++i)
		{
			s2Body* body = bodies + i;
			if (s2IsFree(&body->object))
			{
				continue;
			}

			if (body->type == s2_staticBody)
			{
				continue;
			}

			float invMass = body->invMass;
			float invI = body->invI;

			s2Vec2 v = body->linearVelocity;
			float w = body->angularVelocity;

			// integrate velocities
			v = s2Add(v, s2MulSV(h * invMass, s2MulAdd(body->force, body->mass * body->gravityScale, gravity)));
			w = w + h * invI * body->torque;

			// damping
			v = s2MulSV(1.0f / (1.0f + h * body->linearDamping), v);
			w *= 1.0f / (1.0f + h * body->angularDamping);

			body->linearVelocity = v;
			body->angularVelocity = w;

			// store previous rotation
			body->rot0 = body->rot;

			// integrate positions
			// this is unique to XPBD, no other solvers update position immediately
			body->deltaPosition0 = body->deltaPosition;
			body->deltaPosition = s2MulAdd(body->deltaPosition, h, v);
			body->rot = s2IntegrateRot(body->rot, h * w);
		}

		for (int i = 0; i < jointCapacity; ++i)
		{
			s2Joint* joint = joints + i;
			if (s2IsFree(&joint->object))
			{
				continue;
			}

			s2SolveJoint_XPBD(joint, context, inv_h);
		}

		s2SolveContactPositions_XPBD(world, constraints, constraintCount, h);

		// Project velocities
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

			body->linearVelocity0 = body->linearVelocity;
			body->angularVelocity0 = body->angularVelocity;

			body->linearVelocity = s2MulSV(inv_h, s2Sub(body->deltaPosition, body->deltaPosition0));

			if (s2Length(body->linearVelocity) > 10.0f)
			{
				body->linearVelocity.x += 0.0f;
			}

			body->angularVelocity = s2ComputeAngularVelocity(body->rot0, body->rot, inv_h);
		}

		// Relax contact velocities
		s2SolveContactVelocities_XPBD(world, constraints, constraintCount, h);
	}
	*/

	// Finalize body position
	// body loop
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

		//body->position = s2Add(body->position, body->deltaPosition);
		//body->deltaPosition = s2Vec2_zero;
	}

	// warm starting is not used, this is just for reporting
	// constraint loop
	/*
	for (int i = 0; i < constraintCount; ++i)
	{
		s2ContactConstraint* constraint = constraints + i;
		s2Contact* contact = constraint->contact;
		s2Manifold* manifold = &contact->manifold;

		for (int j = 0; j < constraint->pointCount; ++j)
		{
			manifold->points[j].normalImpulse = 0;
			 //constraint->points[j].normalImpulse* inv_h;
			manifold->points[j].tangentImpulse = 0;
			 //constraint->points[j].tangentImpulse* inv_h;
		}
	}
	*/

	s2FreeStackItem(world->stackAllocator, rows);
}
