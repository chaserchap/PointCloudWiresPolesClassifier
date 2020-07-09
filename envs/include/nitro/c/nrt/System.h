/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * NITRO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program; if not, If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __NRT_SYSTEM_H__
#define __NRT_SYSTEM_H__

#include "nrt/Defines.h"
#include "nrt/Types.h"
#include "nrt/Error.h"
#include "nrt/Memory.h"
#include "nrt/DLL.h"
#include "nrt/Sync.h"
#include "nrt/Directory.h"
#include "nrt/IOHandle.h"

NRTPROT(nrt_Uint16) nrt_System_swap16(nrt_Uint16 ins);
NRTPROT(nrt_Uint32) nrt_System_swap32(nrt_Uint32 inl);
NRTPROT(nrt_Uint64) nrt_System_swap64c(nrt_Uint64 inl);
NRTPROT(nrt_Uint64) nrt_System_swap64(nrt_Uint64 inl);

#endif
