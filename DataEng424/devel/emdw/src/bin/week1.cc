/*
 * Author     :  (DSP Group, E&E Eng, US)
 * Created on :
 * Copyright  : University of Stellenbosch, all rights retained
 */

// patrec headers
#include "prlite_logging.hpp"  // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "emdw.hpp"
#include "discretetable.hpp"

// standard headers
#include <iostream>  // cout, endl, flush, cin, cerr
#include <cctype>  // toupper
#include <string>  // string
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <random>

using namespace std;
using namespace emdw;

//##################################################################
// Some example code. To compile this, go to the emdw/build
// directory and do a:
// cmake ../; make -j7 example
// To run this while in the build directory, do a:
// src/pmr/example
//
// For your own stuff, make a copy of this one to start with. Then
// edit the CMakeLists.txt (also in this directory) by adding your
// new target in the same way as this example.
//##################################################################

int main(int, char *argv[]) {

  // NOTE: this activates logging and unit tests
  initLogging(argv[0]);
  prlite::TestCase::runAllTests();

  try {

    //*********************************************************
    // Some random generator seeding. Just keep this as is
    //*********************************************************

    unsigned seedVal = emdw::randomEngine.getSeedVal();
    cout <<  seedVal << endl;
    emdw::randomEngine.setSeedVal(seedVal);

    //*********************************************************
    // Predefine some types and constants
    //*********************************************************

    typedef int T;                  // The type of the values that the RVs can take on
    typedef DiscreteTable<T> DT;    // DT now is a short-hand for DiscreteTable<int>
    double defProb = 0.0;           // Any unspecified probs will default to this.
    rcptr< vector<T> > binDom (     // Lists the values that a particular RV can take on
        new vector<T>{0,1});
    rcptr< vector<T> > zDom(
      new vector<T>{0,1,2});

    //*********************************************************
    // Define the RVs
    //*********************************************************

    // The enum statement here predefines two RV ids: the id of X is 0
    // and the id of Y is 1. This is easy enough in very simple
    // problems, for more complex situations involving many RVs this
    // becomes cumbersome and we will need a datastructure such as a
    // map to save the RV ids in. Consult the userguide for more on
    // this.

    enum{X, Y, Z};



    // PRAC
    rcptr<Factor> ptrX = 
      uniqptr<DT>(
        new DT(
          {X},
          {binDom},
          defProb,
          {
            {{0}, 0.5},
            {{1}, 0.5},
      } ));

    std::cout << __FILE__ << __LINE__ << ": " << *ptrX << std::endl;

    rcptr<Factor> ptrY = 
      uniqptr<DT>(
        new DT(
          {Y},
          {binDom},
          defProb,
          {
            {{0}, 0.5},
            {{1}, 0.5},
      } ));

    std::cout << __FILE__ << __LINE__ << ": " << *ptrY << std::endl;

    rcptr<Factor> ptrXYZ = 
      uniqptr<DT>(
        new DT(
          {X,Y,Z},
          {binDom,binDom,zDom},
          defProb,
          {
            {{0, 0, 0}, 1},
            {{0, 1, 1}, 1},
            {{1, 0, 1}, 1
            
            {{1, 1, 2}, 1}
      } ));

    std::cout << __FILE__ << __LINE__ << ": " << *ptrXYZ << std::endl;


  // Joint distribution
  rcptr<Factor> jointXYZ = ptrX->absorb(ptrY)->absorb(ptrXYZ);
  std::cout << __FILE__ << __LINE__ << ": P(X,Y,Z) = " << *jointXYZ << std::endl;

  // Marginals
  rcptr<Factor> margXZ = ptrXYZ->marginalize({X, Z})->normalize();
  std::cout << __FILE__ << __LINE__ << ": P(X,Z) = "
        << *margXZ << std::endl;

  std::cout << __FILE__ << __LINE__ << ": P(Z) = "
        << *ptrXYZ->marginalize({Z})->normalize() << std::endl;

    //conditional distribution
    std::cout << __FILE__ << __LINE__ << ": " << *margXZ->observeAndReduce({X},{1})->normalize() << std::endl;

    // Marginals
    rcptr<Factor> margZX = ptrXYZ->marginalize({Z, X})->normalize();
    std::cout << __FILE__ << __LINE__ << ": P(Z,X) = " << *margZX << std::endl;

    //conditional distribution
    std::cout << __FILE__ << __LINE__ << ": " << *margZX->observeAndReduce({Z},{2})->normalize() << std::endl;


    return 0; // tell the world that all is fine
  } // try

  catch (char msg[]) {
    cerr << msg << endl;
  } // catch

  // catch (char const* msg) {
  //   cerr << msg << endl;
  // } // catch

  catch (const string& msg) {
    cerr << msg << endl;
    throw;
  } // catch

  catch (const exception& e) {
    cerr << "Unhandled exception: " << e.what() << endl;
    throw e;
  } // catch

  catch(...) {
    cerr << "An unknown exception / error occurred\n";
    throw;
  } // catch

} // main
