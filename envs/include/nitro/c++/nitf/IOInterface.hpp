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

#ifndef __NITF_IO_INTERFACE_HPP__
#define __NITF_IO_INTERFACE_HPP__

#include "nitf/System.hpp"
#include "nitf/Object.hpp"
#include <string>

/*!
 *  \file IOInterface.hpp
 */
namespace nitf
{

struct DLL_PUBLIC_CLASS IOInterfaceDestructor : public nitf::MemoryDestructor<nitf_IOInterface>
{
    ~IOInterfaceDestructor()
    {
    }
    void operator()(nitf_IOInterface *io);
};

/*!
 *  \class IOInterface
 *  \brief  The C++ wrapper for the nitf_IOInterface
 */
class DLL_PUBLIC_CLASS IOInterface : public nitf::Object<nitf_IOInterface, IOInterfaceDestructor>
{
public:
    // Set native object
    IOInterface(nitf_IOInterface * x)
    {
        setNative(x);
        getNativeOrThrow();
    }

    //! Copy constructor
    IOInterface(const IOInterface& lhs)
    {
        setNative(lhs.getNative());
    }

    //! Destructor
    virtual ~IOInterface() { }

    //! Assignment Operator
    IOInterface & operator=(const IOInterface & x)
    {
        if (&x != this)
            setNative(x.getNative());
        return *this;
    }

    void read(char * buf, size_t size);

    void write(const char * buf, size_t size);

    bool canSeek() const;

    nitf::Off seek(nitf::Off offset, int whence);

    nitf::Off tell() const;

    nitf::Off getSize() const;

    int getMode() const;

    void close();

protected:
    mutable nitf_Error error;
};

}
#endif
