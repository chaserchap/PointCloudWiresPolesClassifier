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

#ifndef __NITF_IOHANDLE_HPP__
#define __NITF_IOHANDLE_HPP__

#include <string>

#include "nitf/NITFException.hpp"
#include "nitf/System.hpp"
#include "nitf/IOInterface.hpp"

/*!
 * \file IOHandle.hpp
 * \brief Contains wrapper implementation for IOHandle
 */

namespace nitf
{

/*!
 *  \class IOHandle
 *  \brief The C++ wrapper of the nitf_IOHandle
 */
class DLL_PUBLIC_CLASS IOHandle : public IOInterface
{

public:

    /*!
     * Default constructor. Only here for compatibility purposes.
     * Use the other constructor
     */
    /*explicit IOHandle()
    {
    }*/

    IOHandle(const std::string& fname,
             nitf::AccessFlags access = NITF_ACCESS_READONLY,
             nitf::CreationFlags creation = NITF_OPEN_EXISTING);

    IOHandle(const char* fname,
             nitf::AccessFlags access = NITF_ACCESS_READONLY,
             nitf::CreationFlags creation = NITF_OPEN_EXISTING);

private:
    static
    nitf_IOInterface*
    open(const char* fname,
         nitf::AccessFlags access,
         nitf::CreationFlags creation);

};

}
#endif
