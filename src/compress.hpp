// compressor for cuckatoo nodes where edgetrimming
// has left at most 2^-compressbits nodes in each partition
template <typename word_t>
class compress {
public:
  u32 NODEBITS;
  u32 COMPRESSBITS;
  u32 SIZEBITS;
  u32 SIZEBITS1;
  word_t SIZE;
  word_t MASK;
  word_t MASK1;
  word_t npairs;
  const static word_t NIL = ~(word_t)0;
  word_t *nodes;

  compress(u32 nodebits, u32 compressbits) {
    NODEBITS = nodebits;
    COMPRESSBITS = compressbits;
    SIZEBITS = NODEBITS-COMPRESSBITS;
    SIZEBITS1 = SIZEBITS-1;
    SIZE = (word_t)1 << SIZEBITS;
    nodes = new word_t[SIZE];
    MASK = SIZE-1;
    MASK1 = MASK >> 1;
  }

  ~compress() {
    delete[] nodes;
  }

  void reset() {
    memset(nodes, (char)NIL, sizeof(word_t[SIZE]));
    npairs = 0;
  }

  word_t operator()(word_t u) {
    u32 parity = u & 1;
    word_t ui = u >> COMPRESSBITS;
    u >>= 1;
    for (; ; ui = (ui+1) & MASK) {
      word_t cu = nodes[ui];
      if (cu == NIL) {
        if (npairs >= SIZE/2) {
          printf("NODE OVERFLOW at %x; LOWER REDUCE_NONCES\n", u << 1 | parity);
          return parity;
        }
        nodes[ui] = u << SIZEBITS1 | npairs;
        return (npairs++ << 1) | parity;
      }
      if ((cu & ~MASK1) == u << SIZEBITS1) {
        return ((cu & MASK1) << 1) | parity;
      }
    }
  }
};
