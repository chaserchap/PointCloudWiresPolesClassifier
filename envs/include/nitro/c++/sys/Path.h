/* =========================================================================
 * This file is part of sys-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * sys-c++ is free software; you can redistribute it and/or modify
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
 * License along with this program; If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __SYS_PATH_H__
#define __SYS_PATH_H__

#include "OS.h"
#include "Export.h"

#include <import/str.h>
#include <string>
#include <deque>
#include <utility>

/*!
 *  \file
 *  \brief This class provides Path-related static utilities.
 *
 */

namespace sys
{

class DLL_PUBLIC_CLASS Path
{
public:
    Path();
    Path(const Path& parent, std::string child);
    Path(std::string parent, std::string child);
    Path(std::string pathName);
    Path& operator=(const Path& path);
    Path(const Path& path);
    ~Path();

    inline operator std::string() const
    {
        return mPathName;
    }

    //! Shortcut for a std::pair of std::strings
    typedef std::pair<std::string, std::string> StringPair;

    /*!
     * Normalizes a pathname. Collapses redundant separators and up-level
     * references. On Windows, it converts forward slashes to backward slashes.
     */
    static std::string normalizePath(const std::string& path);

    inline std::string normalize() const
    {
        return normalizePath(mPathName);
    }

    /*!
     * Joins two paths together, using the OS-specific delimiter.
     */
    static std::string joinPaths(const std::string& path1,
                                 const std::string& path2);

    inline Path join(const std::string& path) const
    {
        return joinPaths(mPathName, path);
    }

    /*!
     * Return a normalized absolutized verion of the pathname supplied.
     */
    static std::string absolutePath(const std::string& path);

    inline std::string getAbsolutePath() const
    {
        return absolutePath(mPathName);
    }

    /*!
     *  Separates a path into its components, and returns a vector of
     *  them. This splits on both '/' and '\\'.
     */
    static std::vector<std::string> separate(const std::string& path);

    inline std::vector<std::string> separate() const
    {
        return separate(mPathName);
    }

    /*!
     * Splits the path into two components: head & tail.
     * 
     * The tail part will never contain the delim; if path ends in the delim,
     * tail will be empty. If there is no delim in path, head will be empty.
     */
    static StringPair splitPath(const std::string& path);

    inline StringPair split() const
    {
        return splitPath(mPathName);
    }

    /*!
     * Splits the path it into two components: drive & tail.
     * 
     * For systems that do not support drive specifications, drive will always
     * be the empty string. drive + tail = path
     */
    static StringPair splitDrive(const std::string& path);

    inline StringPair splitDrive() const
    {
        return splitDrive(mPathName);
    }

    /*!
     * Splits the pathname into two components: root & ext.
     * 
     * The returned pair will contain the root of the path plus the extension,
     * if one exists. You can combine the returned pair to create the original
     * path.
     */
    static StringPair splitExt(const std::string& path);

    inline StringPair splitExt() const
    {
        return splitExt(mPathName);
    }

    /*!
     * Returns the base name of the path supplied. This is the second half of the
     * pair returned by splitPath()
     */
    static std::string basename(const std::string& path, bool rmvExt = false);

    inline std::string getBasePath(bool removeExt = false) const
    {
        return sys::Path::basename(mPathName, removeExt);
    }

    /*!
     * Returns the path delimiter
     */
    static const char* delimiter();

    /*!
     * Returns the path separator
     */
    static const char* separator();

    inline bool exists() const
    {
        return mOS.exists(mPathName);
    }

    inline bool isDirectory() const
    {
        return mOS.isDirectory(mPathName);
    }

    inline bool isFile() const
    {
        return mOS.isFile(mPathName);
    }

    inline sys::Off_T lastModified() const
    {
        return mOS.getLastModifiedTime(mPathName);
    }

    inline std::string getPath() const
    {
        return mPathName;
    }

    std::vector<std::string> list() const;

    /*!
     *  Creates the directory 
     *  \param makeParents - flag for making any non-existant parent 
     *                       directories
     */
    inline bool makeDirectory(bool makeParents = false) const
    {
        if (makeParents)
        {
            std::vector<std::string> pathList = separate(mPathName);
            Path workingPath;
            bool createDir = true;
            size_t i = 0;

            // unix puts slash before any path making cwd == root
            if (pathList.size() && pathList[0] == ".")
            {
                workingPath = pathList[0];
                i++;
            }

            while (createDir && i < pathList.size())
            {
                // create each
                workingPath = workingPath.join(pathList[i]);
                if (!mOS.exists(workingPath))
                    createDir = mOS.makeDirectory(workingPath);
                i++;
            }
            return createDir;
        }
        else
            return mOS.makeDirectory(mPathName);
    }

    inline bool remove() const
    {
        return mOS.remove(mPathName);
    }

    inline bool renameTo(std::string dest)
    {
        if (mOS.move(mPathName, dest))
        {
            mPathName = dest;
            return true;
        }
        return false;
    }

    inline sys::Off_T length() const
    {
        return mOS.getSize(mPathName);
    }

    inline void reset(std::string str)
    {
        mPathName = str;
    }

protected:
    std::string mPathName;
    OS mOS;
};

}

std::ostream& operator<<(std::ostream& os, const sys::Path& path);
std::istream& operator>>(std::istream& os, sys::Path& path);

#endif
