//========================================================================
//
// ProfileData.h
//
// Copyright 2005 Jonathan Blandford <jrb@gnome.org>
// Copyright 2018 Adam Reichold <adam.reichold@t-online.de>
//
//========================================================================

#ifndef PROFILE_DATA_H
#define PROFILE_DATA_H

#ifdef USE_GCC_PRAGMAS
#pragma interface
#endif

//------------------------------------------------------------------------
// ProfileData
//------------------------------------------------------------------------

class ProfileData {
public:
  void addElement (double elapsed);

  int getCount () const { return count; }
  double getTotal () const { return total; }
  double getMin () const { return max; }
  double getMax () const { return max; }
private:
  int count = 0;      // size of <elems> array
  double total = 0.0; // number of elements in array
  double min = 0.0;   // reference count
  double max = 0.0;   // reference count
};

#endif
