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
    rcptr< vector<T> > zDom(new vector<T>{0, 1});
    rcptr< vector<T> > yDom(new vector<T>{0, 1, 2, 3, 4, 5});

    //*********************************************************
    // Define the RVs
    //*********************************************************
    enum { Z1, Z2, Z3, Y1, Y2, Y3 };

    //factors

    rcptr<Factor> pZ1 = uniqptr<DT>(new DT(
      {Z1}, {zDom}, defProb, {
        {{0}, 0.5}, // Fair
        {{1}, 0.5}  // Loaded
    }));

    rcptr<Factor> pZ2_Z1 = uniqptr<DT>(new DT(
      {Z1, Z2}, {zDom, zDom}, defProb, {
        {{0, 0}, 0.95}, 
        {{0, 1}, 0.05},
        {{1, 0}, 0.10}, 
        {{1, 1}, 0.90} 
    }));

    rcptr<Factor> pZ3_Z2 = uniqptr<DT>(new DT(
      {Z2, Z3}, {zDom, zDom}, defProb, {
        {{0, 0}, 0.95}, 
        {{0, 1}, 0.05},
        {{1, 0}, 0.10}, 
        {{1, 1}, 0.90}
    }));

    rcptr<Factor> pY1_Z1 = uniqptr<DT>(new DT(
      {Z1, Y1}, {zDom, yDom}, defProb, {
        {{0, 0}, 1.0/6.0}, {{0, 1}, 1.0/6.0}, {{0, 2}, 1.0/6.0},
        {{0, 3}, 1.0/6.0}, {{0, 4}, 1.0/6.0}, {{0, 5}, 1.0/6.0},

        {{1, 0}, 0.1}, {{1, 1}, 0.1}, {{1, 2}, 0.1},
        {{1, 3}, 0.1}, {{1, 4}, 0.1}, {{1, 5}, 0.5}
    }));

    rcptr<Factor> pY2_Z2 = uniqptr<DT>(new DT(
      {Z2, Y2}, {zDom, yDom}, defProb, {
        {{0, 0}, 1.0/6.0}, {{0, 1}, 1.0/6.0}, {{0, 2}, 1.0/6.0},
        {{0, 3}, 1.0/6.0}, {{0, 4}, 1.0/6.0}, {{0, 5}, 1.0/6.0},
        {{1, 0}, 0.1}, {{1, 1}, 0.1}, {{1, 2}, 0.1},
        {{1, 3}, 0.1}, {{1, 4}, 0.1}, {{1, 5}, 0.5}
    }));

    rcptr<Factor> pY3_Z3 = uniqptr<DT>(new DT(
      {Z3, Y3}, {zDom, yDom}, defProb, {
        {{0, 0}, 1.0/6.0}, {{0, 1}, 1.0/6.0}, {{0, 2}, 1.0/6.0},
        {{0, 3}, 1.0/6.0}, {{0, 4}, 1.0/6.0}, {{0, 5}, 1.0/6.0},
        {{1, 0}, 0.1}, {{1, 1}, 0.1}, {{1, 2}, 0.1},
        {{1, 3}, 0.1}, {{1, 4}, 0.1}, {{1, 5}, 0.5}
    }));

    rcptr<Factor> jointHMM = pZ1->absorb(pZ2_Z1)->absorb(pZ3_Z2)->absorb(pY1_Z1)->absorb(pY2_Z2)->absorb(pY3_Z3);

    std::cout << "--- Joint Distribution P(Z1, Z2, Z3, Y1, Y2, Y3) ---\n" << *jointHMM << std::endl;

    //3.1 c
    // with 3 time steps we have 2^3 * 6^3 = 1728 entries.
    //with T time steps we have 2^T * 6^T = 12^T. which get ubsardly large

    //3.1 d

    // We observe: Rolls 6, 5, 6
    rcptr<Factor> evidence = jointHMM->observeAndReduce({Y1, Y2, Y3}, {5, 4, 5});

    // Calculate P(Z1 | evidence)
    rcptr<Factor> postZ1 = evidence->marginalize({Z1})->normalize();
    std::cout << "--- P(Z1 | Y=(6,5,6)) ---\n" << *postZ1 << std::endl;

    // Calculate P(Z2 | evidence)
    rcptr<Factor> postZ2 = evidence->marginalize({Z2})->normalize();
    std::cout << "--- P(Z2 | Y=(6,5,6)) ---\n" << *postZ2 << std::endl;

    // Calculate P(Z3 | evidence)
    rcptr<Factor> postZ3 = evidence->marginalize({Z3})->normalize();
    std::cout << "--- P(Z3 | Y=(6,5,6)) ---\n" << *postZ3 << std::endl;

    //3.2 a
    //if z2 is unobserved - they are dependent
    //if z2 is observed - they are conditionally independent

    //3.2 b
    // *********************************************************
    // 6. Chain Structure: Flow of Influence between Z1 and Z3
    // *********************************************************

    //Z2 is UNOBSERVED
    
    rcptr<Factor> margZ1 = jointHMM->marginalize({Z1})->normalize();
    std::cout << "P(Z1):\n" << *margZ1 << std::endl;

    // Observe Z3 = 1 (Loaded), Z2 unobserved
    rcptr<Factor> margZ1_given_Z3 = jointHMM->observeAndReduce({Z3}, {1})->marginalize({Z1})->normalize();
    std::cout << "P(Z1 | Z3=1):\n" << *margZ1_given_Z3 << std::endl;


    //Z2 is OBSERVED

    // Observe Z2 = 1 (Loaded), Z3 unobserved
    rcptr<Factor> margZ1_given_Z2 = jointHMM->observeAndReduce({Z2}, {1})->marginalize({Z1})->normalize();
    std::cout << "P(Z1 | Z2=1):\n" << *margZ1_given_Z2 << std::endl;

    // Observe Z2 = 1 and Z3 = 1
    rcptr<Factor> margZ1_given_Z2Z3 = jointHMM->observeAndReduce({Z2, Z3}, {1, 1})->marginalize({Z1})->normalize();
    std::cout << "P(Z1 | Z2=1, Z3=1):\n" << *margZ1_given_Z2Z3 << std::endl;

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
