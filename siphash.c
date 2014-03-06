#include <sys/cdefs.h>
#include <strings.h>
#include <sys/param.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
// #include <sys/endian.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <openssl/sha.h> // if openssl absent, you can instead use #include "sha256.c"

typedef struct {
	uint64_t	v[4];
} siphash_ctx;

void SipHash_Init24(SIPHASH_CTX *);
void SipHash_SetKey(SIPHASH_CTX *, const uint8_t [16]);
void SipHash_Update(SIPHASH_CTX *, const void *, size_t);
void SipHash_Final(void *, SIPHASH_CTX *);
uint64_t SipHash_End(SIPHASH_CTX *);
uint64_t SipHash24(SIPHASH_CTX *, int, int, const uint8_t [16], const void *, size_t);

static void SipRounds(SIPHASH_CTX *ctx, int final);

void SipHash_Init24(SIPHASH_CTX *ctx) {
	ctx->buf.b64 = 0;
}

static __inline uint32_t
le32dec(const void *pp) {
uint8_t const *p = (uint8_t const *)pp;
return (((unsigned)p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0]);
}
static __inline uint64_t
le64dec(const void *pp) {
uint8_t const *p = (uint8_t const *)pp;
return (((uint64_t)le32dec(p + 4) << 32) | le32dec(p));
}
#define le64toh(x)      ((uint64_t)(x))
static __inline void
le32enc(void *pp, uint32_t u) {
uint8_t *p = (uint8_t *)pp;
p[0] = u & 0xff;
p[1] = (u >> 8) & 0xff;
p[2] = (u >> 16) & 0xff;
p[3] = (u >> 24) & 0xff;
}
static __inline void
le64enc(void *pp, uint64_t u) {
uint8_t *p = (uint8_t *)pp;
le32enc(p, (uint32_t)(u & 0xffffffffU));
le32enc(p + 4, (uint32_t)(u >> 32));
}


void SipHash_SetKey(SIPHASH_CTX *ctx, const uint8_t key[16]) {
	ctx->v[0] = 0x736f6d6570736575ull;
	ctx->v[1] = 0x646f72616e646f6dull;
	ctx->v[2] = 0x6c7967656e657261ull;
	ctx->v[3] = 0x7465646279746573ull;
	uint64_t k[2];
	k[0] = le64dec(&key[0]);
	k[1] = le64dec(&key[8]);
	ctx->v[0] ^= k[0];
	ctx->v[1] ^= k[1];
	ctx->v[2] ^= k[0];
	ctx->v[3] ^= k[1];
}

static size_t SipBuf(SIPHASH_CTX *ctx, const uint8_t **src, size_t len, int final) {
	size_t x = 0;
	if (!final) {
		x = MIN(len, sizeof(ctx->buf.b64) - ctx->buflen);
		bcopy(*src, &ctx->buf.b8[ctx->buflen], x);
		ctx->buflen += x;
		*src += x;
	} else
		ctx->buf.b8[7] = (uint8_t)ctx->bytes;
	if (ctx->buflen == 8 || final) {
		ctx->v[3] ^= le64toh(ctx->buf.b64);
		SipRounds(ctx, 0);
		ctx->v[0] ^= le64toh(ctx->buf.b64);
		ctx->buf.b64 = 0;
		ctx->buflen = 0;
	}
	return (x);
}

void SipHash_Update(SIPHASH_CTX *ctx, const void *src, size_t len) {
	uint64_t m;
	const uint64_t *p;
	const uint8_t *s;
	size_t rem;
	s = src;
	ctx->bytes += len;
	SipBuf(ctx, &s, len, 0);
}

void SipHash_Final(void *dst, SIPHASH_CTX *ctx) {
	uint64_t r;
	r = SipHash_End(ctx);
	le64enc(dst, r);
}

uint64_t SipHash_End(SIPHASH_CTX *ctx) {
	uint64_t r;
	SipBuf(ctx, NULL, 0, 1);
	ctx->v[2] ^= 0xff;
	SipRounds(ctx, 1);
	r = (ctx->v[0] ^ ctx->v[1]) ^ (ctx->v[2] ^ ctx->v[3]);
	bzero(ctx, sizeof(*ctx));
	return (r);
}

uint64_t SipHash24(SIPHASH_CTX *ctx, int rc, int rf, const uint8_t key[16], const void *src, size_t len) {
	SipHash_Init24(ctx);
	SipHash_SetKey(ctx, key);
	SipHash_Update(ctx, src, len);
	return (SipHash_End(ctx));
}

#define SIP_ROTL(x, b)	(uint64_t)(((x) << (b)) | ( (x) >> (64 - (b))))

static void SipRounds(SIPHASH_CTX *ctx, int final) {
	int rounds = final ? 4 : 2;
	while (rounds--) {
		ctx->v[0] += ctx->v[1];
		ctx->v[2] += ctx->v[3];
		ctx->v[1] = SIP_ROTL(ctx->v[1], 13);
		ctx->v[3] = SIP_ROTL(ctx->v[3], 16);

		ctx->v[1] ^= ctx->v[0];
		ctx->v[3] ^= ctx->v[2];
		ctx->v[0] = SIP_ROTL(ctx->v[0], 32);

		ctx->v[2] += ctx->v[1];
		ctx->v[0] += ctx->v[3];
		ctx->v[1] = SIP_ROTL(ctx->v[1], 17);
		ctx->v[3] = SIP_ROTL(ctx->v[3], 21);

		ctx->v[1] ^= ctx->v[2];
		ctx->v[3] ^= ctx->v[0];
		ctx->v[2] = SIP_ROTL(ctx->v[2], 32);
	}
}


// proof-of-work parameters
#ifndef SIZEMULT 
#define SIZEMULT 1
#endif
#ifndef SIZESHIFT 
#define SIZESHIFT 20
#endif
#ifndef EASINESS 
#define EASINESS (SIZE/2)
#endif
#ifndef PROOFSIZE 
#define PROOFSIZE 42
#endif
// algorithm parameters
#ifndef NTHREADS
#define NTHREADS 1
#endif

#define SIZE (SIZEMULT*((unsigned)1<<SIZESHIFT))
// relatively prime partition sizes
#define PARTU (SIZE/2+1)
#define PARTV (SIZE/2-1)

typedef uint64_t u64;
 
#define ROTL(x,b) (u64)( ((x) << (b)) | ( (x) >> (64 - (b))) )
 
#define SIPROUND \
  do { \
    v0 += v1; v1=ROTL(v1,13); v1 ^= v0; v0=ROTL(v0,32); \
    v2 += v3; v3=ROTL(v3,16); v3 ^= v2; \
    v0 += v3; v3=ROTL(v3,21); v3 ^= v0; \
    v2 += v1; v1=ROTL(v1,17); v1 ^= v2; v2=ROTL(v2,32); \
  } while(0)
 
// SipHash-2-4 specialized to precomputed key and 4 byte nonces
u64 siphash24( unsigned nonce, u64 v0, u64 v1, u64 v2, u64 v3) {
  u64 b = ( ( u64 )4 ) << 56 | nonce;
  v3 ^= b;
  SIPROUND; SIPROUND;
  v0 ^= b;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return v0 ^ v1 ^ v2  ^ v3;
}

u64 v0 = 0x736f6d6570736575ULL, v1 = 0x646f72616e646f6dULL,
    v2 = 0x6c7967656e657261ULL, v3 = 0x7465646279746573ULL;

#define U8TO64_LE(p) \
  (((u64)((p)[0])      ) | ((u64)((p)[1]) <<  8) | \
   ((u64)((p)[2]) << 16) | ((u64)((p)[3]) << 24) | \
   ((u64)((p)[4]) << 32) | ((u64)((p)[5]) << 40) | \
   ((u64)((p)[6]) << 48) | ((u64)((p)[7]) << 56))
 
// derive siphash key from header
void setheader(char *header) {
  unsigned char hdrkey[32];
  SHA256((unsigned char *)header, strlen(header), hdrkey);
  u64 k0 = U8TO64_LE( hdrkey ); u64 k1 = U8TO64_LE( hdrkey + 8 );
  v3 ^= k1; v2 ^= k0; v1 ^= k1; v0 ^= k0;
}

u64 siphash(unsigned nonce) {
  return siphash24(nonce, v0, v1, v2, v3);
}

// generate edge in cuckoo graph
void sipedge(unsigned nonce, unsigned *pu, unsigned *pv) {
  u64 sip = siphash(nonce);
  *pu = 1 +         (unsigned)(sip % PARTU);
  *pv = 1 + PARTU + (unsigned)(sip % PARTV);
}
