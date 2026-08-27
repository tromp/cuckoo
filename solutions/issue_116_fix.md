**Patch Summary**

The original `siphash.c` implementation incorrectly handled the final partial block when the input length was not a multiple of 8 bytes. This caused incorrect hash values for many inputs, which in turn broke the SipHash‑1‑2 bounty implementation.  
The patch replaces the faulty final‑block logic with a correct, standards‑compliant implementation that:

1. Pads the last block with zeros.
2. Stores the original length in the last byte of the padded block.
3. Performs the final SipHash rounds exactly as specified in the SipHash paper.

The change is fully self‑contained, adds no new dependencies, and keeps the public API unchanged.

---

**Diff**

```diff
--- a/src/siphash.c
+++ b/src/siphash.c
@@
 #include <stdint.h>
 #include <stddef.h>
 
 /* SipHash-1-2 implementation */
 
-/* Original