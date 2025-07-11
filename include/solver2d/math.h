// SPDX-FileCopyrightText: 2024 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"

#include <math.h>

#define S2_MIN(A, B) ((A) < (B) ? (A) : (B))
#define S2_MAX(A, B) ((A) > (B) ? (A) : (B))
#define S2_ABS(A) ((A) > 0.0f ? (A) : -(A))
#define S2_CLAMP(A, B, C) S2_MIN(S2_MAX(A, B), C)

static const s2Vec2 s2Vec2_zero = {0.0f, 0.0f};
static const s2Rot s2Rot_identity = {0.0f, 1.0f};
static const s2Transform s2Transform_identity = {{0.0f, 0.0f}, {0.0f, 1.0f}};
static const s2Mat22 s2Mat22_zero = {{0.0f, 0.0f}, {0.0f, 0.0f}};

#ifdef __cplusplus
extern "C"
{
#endif

bool s2IsValid(float a);
bool s2IsValidVec2(s2Vec2 v);

/// Convert this vector into a unit vector
s2Vec2 s2Normalize(s2Vec2 v);

/// This asserts of the vector is too short
s2Vec2 s2NormalizeChecked(s2Vec2 v);

s2Vec2 s2GetLengthAndNormalize(float* length, s2Vec2 v);

#ifdef __cplusplus
}
#endif

/// Make a vector
static inline s2Vec2 s2MakeVec2(float x, float y)
{
	return S2_LITERAL(s2Vec2){x, y};
}

/// Make a vector
static inline s2Vec3 s2MakeVec3(float x, float y, float z)
{
	return S2_LITERAL(s2Vec3){x, y, z};
}

/// Vector dot product
static inline float s2Dot(s2Vec2 a, s2Vec2 b)
{
	return a.x * b.x + a.y * b.y;
}

/// Vector dot product
static inline float s2Dot3(s2Vec3 a, s2Vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Vector cross product. In 2D this yields a scalar.
static inline float s2Cross(s2Vec2 a, s2Vec2 b)
{
	return a.x * b.y - a.y * b.x;
}

/// Perform the cross product on a vector and a scalar. In 2D this produces
/// a vector.
static inline s2Vec2 s2CrossVS(s2Vec2 v, float s)
{
	return S2_LITERAL(s2Vec2){s * v.y, -s * v.x};
}

/// Perform the cross product on a scalar and a vector. In 2D this produces
/// a vector.
static inline s2Vec2 s2CrossSV(float s, s2Vec2 v)
{
	return S2_LITERAL(s2Vec2){-s * v.y, s * v.x};
}

/// Get a right pointing perpendicular vector. Equivalent to s2CrossVS(v, 1.0f).
static inline s2Vec2 s2RightPerp(s2Vec2 v)
{
	return S2_LITERAL(s2Vec2){v.y, -v.x};
}

/// Get a left pointing perpendicular vector. Equivalent to b2CrossSV(1.0f, v)
static inline s2Vec2 s2LeftPerp(s2Vec2 v)
{
	return S2_LITERAL(s2Vec2){-v.y, v.x};
}

/// Vector addition
static inline s2Vec2 s2Add(s2Vec2 a, s2Vec2 b)
{
	return S2_LITERAL(s2Vec2){a.x + b.x, a.y + b.y};
}

/// Vector subtraction
static inline s2Vec2 s2Sub(s2Vec2 a, s2Vec2 b)
{
	return S2_LITERAL(s2Vec2){a.x - b.x, a.y - b.y};
}

/// Vector subtraction
static inline s2Vec3 s2Sub3(s2Vec3 a, s2Vec3 b)
{
	return S2_LITERAL(s2Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

/// Vector subtraction
static inline s2Vec2 s2Neg(s2Vec2 a)
{
	return S2_LITERAL(s2Vec2){-a.x, -a.y};
}

/// Vector linear interpolation
static inline s2Vec2 s2Lerp(s2Vec2 a, s2Vec2 b, float t)
{
	return S2_LITERAL(s2Vec2){a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
}

/// Component-wise multiplication
static inline s2Vec2 s2Mul(s2Vec2 a, s2Vec2 b)
{
	return S2_LITERAL(s2Vec2){a.x * b.x, a.y * b.y};
}

/// Component-wise multiplication
static inline s2Vec3 s2Mul3(s2Vec3 a, s2Vec3 b)
{
	return S2_LITERAL(s2Vec3){a.x * b.x, a.y * b.y, a.z * b.z};
}

/// Multiply a scalar and vector
static inline s2Vec2 s2MulSV(float s, s2Vec2 v)
{
	return S2_LITERAL(s2Vec2){s * v.x, s * v.y};
}

static inline s2Vec3 s2MulSV3(float s, s2Vec3 v)
{
	return S2_LITERAL(s2Vec3){s * v.x, s * v.y, s * v.z};
}

/// a + s * b
static inline s2Vec2 s2MulAdd(s2Vec2 a, float s, s2Vec2 b)
{
	return S2_LITERAL(s2Vec2){a.x + s * b.x, a.y + s * b.y};
}

/// a + s * b
static inline s2Vec3 s2MulAdd3(s2Vec3 a, float s, s2Vec3 b)
{
	return S2_LITERAL(s2Vec3){a.x + s * b.x, a.y + s * b.y, a.z + s * b.z};
}

/// a - s * b
static inline s2Vec2 s2MulSub(s2Vec2 a, float s, s2Vec2 b)
{
	return S2_LITERAL(s2Vec2){a.x - s * b.x, a.y - s * b.y};
}

/// a + s * b * b^T
static inline s2Mat33 s2AddScaledOuter(s2Mat33 a, float s, s2Vec3 b)
{
	s2Mat33 c;
	c.cx.x = a.cx.x + s * b.x * b.x;
	c.cx.y = a.cx.y + s * b.x * b.y;
	c.cx.z = a.cx.z + s * b.x * b.z;
	c.cy.x = a.cy.x + s * b.y * b.x;
	c.cy.y = a.cy.y + s * b.y * b.y;
	c.cy.z = a.cy.z + s * b.y * b.z;
	c.cz.x = a.cz.x + s * b.z * b.x;
	c.cz.y = a.cz.y + s * b.z * b.y;
	c.cz.z = a.cz.z + s * b.z * b.z;
	return c;
}

/// Component-wise absolute vector
static inline s2Vec2 s2Abs(s2Vec2 a)
{
	s2Vec2 b;
	b.x = S2_ABS(a.x);
	b.y = S2_ABS(a.y);
	return b;
}

/// Component-wise absolute vector
static inline s2Vec2 s2Min(s2Vec2 a, s2Vec2 b)
{
	s2Vec2 c;
	c.x = S2_MIN(a.x, b.x);
	c.y = S2_MIN(a.y, b.y);
	return c;
}

/// Component-wise absolute vector
static inline s2Vec2 s2MinSV(s2Vec2 a, float b)
{
	s2Vec2 c;
	c.x = S2_MIN(a.x, b);
	c.y = S2_MIN(a.y, b);
	return c;
}

/// Component-wise absolute vector
static inline s2Vec2 s2Max(s2Vec2 a, s2Vec2 b)
{
	s2Vec2 c;
	c.x = S2_MAX(a.x, b.x);
	c.y = S2_MAX(a.y, b.y);
	return c;
}

/// Component-wise clamp vector so v into the range [a, b]
static inline s2Vec2 s2Clamp(s2Vec2 v, s2Vec2 a, s2Vec2 b)
{
	s2Vec2 c;
	c.x = S2_CLAMP(v.x, a.x, b.x);
	c.y = S2_CLAMP(v.y, a.y, b.y);
	return c;
}

/// Component-wise clamp vector so v into the range [a, b]
static inline s2Vec2 s2ClampSV(s2Vec2 v, float a, float b)
{
	s2Vec2 c;
	c.x = S2_CLAMP(v.x, a, b);
	c.y = S2_CLAMP(v.y, a, b);
	return c;
}

/// Get the length of this vector (the norm).
static inline float s2Length(s2Vec2 v)
{
	return sqrtf(v.x * v.x + v.y * v.y);
}

/// Get the length of this vector (the norm).
static inline float s2LengthSquared(s2Vec2 v)
{
	return v.x * v.x + v.y * v.y;
}

static inline float s2Distance(s2Vec2 a, s2Vec2 b)
{
	float dx = b.x - a.x;
	float dy = b.y - a.y;
	return sqrtf(dx * dx + dy * dy);
}

/// Get the length of this vector (the norm).
static inline float s2DistanceSquared(s2Vec2 a, s2Vec2 b)
{
	s2Vec2 c = {b.x - a.x, b.y - a.y};
	return c.x * c.x + c.y * c.y;
}

/// Set using an angle in radians.
static inline s2Rot s2MakeRot(float angle)
{
	s2Rot q = {sinf(angle), cosf(angle)};
	return q;
}

static inline s2Rot s2NormalizeRot(s2Rot q)
{
	float mag = sqrtf(q.s * q.s + q.c * q.c);
	float invMag = mag > 0.0 ? 1.0f / mag : 0.0f;
	s2Rot qn = {q.s * invMag, q.c * invMag};
	return qn;
}

static inline s2Rot s2IntegrateRot(s2Rot q1, float omegah)
{
	// ds/dt = omega * cos(t)
	// dc/dt = -omega * sin(t)
	// s2 = s1 + omega * h * c1
	// c2 = c1 - omega * h * s1
	s2Rot q2 = {q1.s + omegah * q1.c, q1.c - omegah * q1.s};
	return s2NormalizeRot(q2);

	// quaternion multiplication
	// q1 * q2 = {cross(q1.v, q2.v) + q2.v * q1.s + q1.v * q2.s, q1.s * q2.s - dot(q1.v, q2.v)}
	// in 2d this reduces to
	// q1 * q2 = {q2.z * q1.w + q1.z * q2.w, q1.w * q2.w - q1.z * q2.z}

	// integration
	// q2 = q1 + 0.5 * {omegah, 0} * q1
	// = q1 + 0.5 * {omegah * q1.w, -omegah * q1.z}
	// this is identical to the trig version above
	// conclusion: no reason to use 2d quaternions, instead use sine and cosine
	// with explicit integration
}

static inline float s2ComputeAngularVelocity(s2Rot q1, s2Rot q2, float inv_h)
{
	// dc/dt = -omega * sin(t)
	// ds/dt = omega * cos(t)
	// c2 = c1 - omega * h * s1
	// s2 = s1 + omega * h * c1

	// omega * h * s1 = c1 - c2
	// omega * h * c1 = s2 - s1
	// omega * h = (c1 - c2) * s1 + (s2 - s1) * c1;
	// omega * h = s1 * c1 - c2 * s1 + s2 * c1 - s1 * c1
	// omega * h = s2 * c1 - c2 * s1 = sin(a2 - a1) ~= a2 - a1 for small delta
	float omega = inv_h * (q2.s * q1.c - q2.c * q1.s);
	return omega;

	// quaternion multiplication
	// q1 * q2 = {cross(q1.v, q2.v) + q2.v * q1.s + q1.v * q2.s, q1.s * q2.s - dot(q1.v, q2.v)}
	// in 2d this reduces to
	// q1 * q2 = {q2.z * q1.w + q1.z * q2.w, q1.w * q2.w - q1.z * q2.z}

	// integration
	// q2 = q1 + 0.5 * {omegah, 0} * q1
	// = q1 + 0.5 * {omegah * q1.w, -omegah * q1.z}
	// this is identical to the trig version above
	// conclusion: no reason to use 2d quaternions, instead use sine and cosine
	// with explicit integration
}

/// Get the angle in radians
static inline float s2Rot_GetAngle(s2Rot q)
{
	return atan2f(q.s, q.c);
}

/// Get the x-axis
static inline s2Vec2 s2Rot_GetXAxis(s2Rot q)
{
	s2Vec2 v = {q.c, q.s};
	return v;
}

/// Get the y-axis
static inline s2Vec2 s2Rot_GetYAxis(s2Rot q)
{
	s2Vec2 v = {-q.s, q.c};
	return v;
}

/// Multiply two rotations: b * a
/// equivalent to angle addition via trig identity:
///	sin(b + a) = sin(b) * cos(a) + cos(b) * sin(a)
///	cos(b + a) = cos(b) * cos(a) - sin(b) * sin(a)
///	order independent!
static inline s2Rot s2MulRot(s2Rot b, s2Rot a)
{
	// [bc -bs] * [ac -as] = [bc*ac-bs*as -bc*as-bs*ac]
	// [bs  bc]   [as  ac]   [bs*ac+bc*as -bs*as+bc*ac]
	// s = bs * ac + bc * as
	// c = bc * ac - bs * as
	s2Rot ba;
	ba.s = b.s * a.c + b.c * a.s;
	ba.c = b.c * a.c - b.s * a.s;
	return ba;
}

/// Transpose multiply two rotations: inv(b) * a
/// equivalent to angle subtraction via trig identity:
///	sin(a - b) = sin(b) * cos(a) - cos(b) * sin(a)
///	cos(a - b) = cos(b) * cos(a) + sin(b) * sin(a)
static inline s2Rot s2InvMulRot(s2Rot b, s2Rot a)
{
	// [ bc bs] * [ac -as] = [ bc*ac+bs*as -bc*as+bs*ac]
	// [-bs bc]   [as  ac]   [-bs*ac+bc*as  bs*as+bc*ac]
	// s = bc * as - bs * ac
	// c = bc * ac + bs * as
	s2Rot bTa;
	bTa.s = b.c * a.s - b.s * a.c;
	bTa.c = b.c * a.c + b.s * a.s;
	return bTa;
}

// relative angle between b and a (rot_b * inv(rot_a))
static inline float s2RelativeAngle(s2Rot b, s2Rot a)
{
	// sin(b - a) = bs * ac - bc * as
	// cos(b - a) = bc * ac + bs * as
	float s = b.s * a.c - b.c * a.s;
	float c = b.c * a.c + b.s * a.s;
	return atan2f(s, c);
}

/// Rotate a vector
static inline s2Vec2 s2RotateVector(s2Rot q, s2Vec2 v)
{
	// 2d quaternion
	// s2CrossSV(s,v) = {-s * v.y, s * v.x}
	// v + 2.0f * s2CrossSV(q.z, s2CrossSV(q.z, v) + q.w * v)
	// v + 2.0f * s2CrossSV(q.z, {-q.z * v.y, q.z * v.x} + q.w * v)
	// v + 2.0f * s2CrossSV(q.z, {-q.z * v.y + q.w * v.x, q.z * v.x + q.w * v.y})
	// v + 2.0f * {-q.z * (q.z * v.x + q.w * v.y), q.z * (-q.z * v.y + q.w * v.x)}
	// {v.x - 2.0f * q.z * (q.z * v.x + q.w * v.y), v.y - 2.0f * q.z * (q.z * v.y - q.w * v.x)}

	return S2_LITERAL(s2Vec2){q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y};
}

/// Inverse rotate a vector
static inline s2Vec2 s2InvRotateVector(s2Rot q, s2Vec2 v)
{
	return S2_LITERAL(s2Vec2){q.c * v.x + q.s * v.y, -q.s * v.x + q.c * v.y};
}

/// Transform a point (e.g. local space to world space)
static inline s2Vec2 s2TransformPoint(s2Transform xf, const s2Vec2 p)
{
	float x = (xf.q.c * p.x - xf.q.s * p.y) + xf.p.x;
	float y = (xf.q.s * p.x + xf.q.c * p.y) + xf.p.y;

	return S2_LITERAL(s2Vec2){x, y};
}

// Inverse transform a point (e.g. world space to local space)
static inline s2Vec2 s2InvTransformPoint(s2Transform xf, const s2Vec2 p)
{
	float vx = p.x - xf.p.x;
	float vy = p.y - xf.p.y;
	return S2_LITERAL(s2Vec2){xf.q.c * vx + xf.q.s * vy, -xf.q.s * vx + xf.q.c * vy};
}

// v2 = A.q.Rot(B.q.Rot(v1) + B.p) + A.p
//    = (A.q * B.q).Rot(v1) + A.q.Rot(B.p) + A.p
static inline s2Transform s2MulTransforms(s2Transform A, s2Transform B)
{
	s2Transform C;
	C.q = s2MulRot(A.q, B.q);
	C.p = s2Add(s2RotateVector(A.q, B.p), A.p);
	return C;
}

// v2 = A.q' * (B.q * v1 + B.p - A.p)
//    = A.q' * B.q * v1 + A.q' * (B.p - A.p)
static inline s2Transform s2InvMulTransforms(s2Transform A, s2Transform B)
{
	s2Transform C;
	C.q = s2InvMulRot(A.q, B.q);
	C.p = s2InvRotateVector(A.q, s2Sub(B.p, A.p));
	return C;
}

static inline s2Vec2 s2MulMV(s2Mat22 A, s2Vec2 v)
{
	s2Vec2 u = {A.cx.x * v.x + A.cy.x * v.y, A.cx.y * v.x + A.cy.y * v.y};
	return u;
}

static inline s2Mat22 s2GetInverse22(s2Mat22 A)
{
	float a = A.cx.x, b = A.cy.x, c = A.cx.y, d = A.cy.y;
	s2Mat22 B;
	float det = a * d - b * c;
	if (det != 0.0f)
	{
		det = 1.0f / det;
	}
	B.cx.x = det * d;
	B.cy.x = -det * b;
	B.cx.y = -det * c;
	B.cy.y = det * a;
	return B;
}

/// Solve A * x = b, where b is a column vector. This is more efficient
/// than computing the inverse in one-shot cases.
static inline s2Vec2 s2Solve22(s2Mat22 A, s2Vec2 b)
{
	float a11 = A.cx.x, a12 = A.cy.x, a21 = A.cx.y, a22 = A.cy.y;
	float det = a11 * a22 - a12 * a21;
	if (det != 0.0f)
	{
		det = 1.0f / det;
	}
	s2Vec2 x = {det * (a22 * b.x - a12 * b.y), det * (a11 * b.y - a21 * b.x)};
	return x;
}
