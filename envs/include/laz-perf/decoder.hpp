/*
===============================================================================

  FILE:  decoder.hpp

  CONTENTS:
    Decoder stuff

  PROGRAMMERS:

    martin.isenburg@rapidlasso.com  -  http://rapidlasso.com
    uday.karan@gmail.com - Hobu, Inc.

  COPYRIGHT:

    (c) 2007-2014, martin isenburg, rapidlasso - tools to catch reality
    (c) 2014, Uday Verma, Hobu, Inc.

    This is free software; you can redistribute and/or modify it under the
    terms of the GNU Lesser General Licence as published by the Free Software
    Foundation. See the COPYING file for more information.

    This software is distributed WITHOUT ANY WARRANTY and without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  CHANGE HISTORY:

===============================================================================
*/

#ifndef __decoder_hpp__
#define __decoder_hpp__

#include <cassert>

#include "common/types.hpp"

namespace laszip {
	namespace decoders {
		template<
			typename TInputStream
		>
		struct arithmetic {
			arithmetic(TInputStream& in) :
				instream(in), value(0) {
				length = AC__MaxLength;
			}

			~arithmetic() {
			}

			void readInitBytes() {
				value =
					(instream.getByte() << 24) |
					(instream.getByte() << 16) |
					(instream.getByte() << 8) |
					instream.getByte();
			}

			template<typename TEntropyModel>
			U32 decodeBit(TEntropyModel& m) {
				U32 x = m.bit_0_prob * (length >> BM__LengthShift);       // product l x p0
				U32 sym = (value >= x);                                          // decision
				// update & shift interval
				if (sym == 0) {
					length  = x;
					++m.bit_0_count;
				}
				else {
					value  -= x;                                  // shifted interval base = 0
					length -= x;
				}

				if (length < AC__MinLength) renorm_dec_interval();        // renormalization
				if (--m.bits_until_update == 0) m.update();       // periodic model update

				return sym;                                         // return data bit value
			}

			template<typename TEntropyModel>
			U32 decodeSymbol(TEntropyModel& m) {
				U32 n, sym, x, y = length;

				if (m.decoder_table) {             // use table look-up for faster decoding
					unsigned dv = value / (length >>= DM__LengthShift);
					unsigned t = dv >> m.table_shift;

					sym = m.decoder_table[t];      // initial decision based on table look-up
					n = m.decoder_table[t+1] + 1;

					while (n > sym + 1) {                      // finish with bisection search
						U32 k = (sym + n) >> 1;
						if (m.distribution[k] > dv) n = k; else sym = k;
					}

					// compute products
					x = m.distribution[sym] * length;
					if (sym != m.last_symbol) y = m.distribution[sym+1] * length;
				}
				else {                                  // decode using only multiplications
					x = sym = 0;
					length >>= DM__LengthShift;
					U32 k = (n = m.symbols) >> 1;
					// decode via bisection search
					do {
						U32 z = length * m.distribution[k];
						if (z > value) {
							n = k;
							y = z;                                             // value is smaller
						}
						else {
							sym = k;
							x = z;                                     // value is larger or equal
						}
					} while ((k = (sym + n) >> 1) != sym);
				}

				value -= x;                                               // update interval
				length = y - x;

				if (length < AC__MinLength) renorm_dec_interval();        // renormalization

				++m.symbol_count[sym];
				if (--m.symbols_until_update == 0) m.update();    // periodic model update

				return sym;
			}

			U32 readBit() {
				U32 sym = value / (length >>= 1);            // decode symbol, change length
				value -= length * sym;                                    // update interval

				if (length < AC__MinLength) renorm_dec_interval();        // renormalization

				return sym;
			}

			U32 readBits(U32 bits) {
				assert(bits && (bits <= 32));

				if (bits > 19) {
					U32 tmp = readShort();
					bits = bits - 16;
					U32 tmp1 = readBits(bits) << 16;
					return (tmp1|tmp);
				}

				U32 sym = value / (length >>= bits);// decode symbol, change length
				value -= length * sym;                                    // update interval

				if (length < AC__MinLength) renorm_dec_interval();        // renormalization
				return sym;
			}

			U8 readByte() {
				U32 sym = value / (length >>= 8);            // decode symbol, change length
				value -= length * sym;                                    // update interval

				if (length < AC__MinLength) renorm_dec_interval();        // renormalization

				assert(sym < (1<<8));

				return (U8)sym;
			}

			U16 readShort() {
				U32 sym = value / (length >>= 16);           // decode symbol, change length
				value -= length * sym;                                    // update interval

				if (length < AC__MinLength) renorm_dec_interval();        // renormalization

				assert(sym < (1<<16));

				return (U16)sym;
			}

			U32 readInt() {
				U32 lowerInt = readShort();
				U32 upperInt = readShort();
				return (upperInt<<16)|lowerInt;
			}

			F32 readFloat() { /* danger in float reinterpretation */
				U32I32F32 u32i32f32;
				u32i32f32.u32 = readInt();
				return u32i32f32.f32;
			}

			U64 readInt64() {
				U64 lowerInt = readInt();
				U64 upperInt = readInt();
				return (upperInt<<32)|lowerInt;
			}

			F64 readDouble() { /* danger in float reinterpretation */
				U64I64F64 u64i64f64;
				u64i64f64.u64 = readInt64();
				return u64i64f64.f64;
			}

			TInputStream& getInStream() {
				return instream;
			}


			arithmetic<TInputStream>(const arithmetic<TInputStream>&) = delete;
			arithmetic<TInputStream>& operator = (const arithmetic<TInputStream>&) = delete;

		private:
			void renorm_dec_interval() {
				do {                                          // read least-significant byte
					value = (value << 8) | instream.getByte();
				} while ((length <<= 8) < AC__MinLength);        // length multiplied by 256
			}

			TInputStream& instream;
			U32 value, length;
		};
	}
}

#endif // __decoder_hpp__
