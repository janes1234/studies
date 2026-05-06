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
    // Domain for D, I, S, L 
    rcptr< vector<T> > dom2(new vector<T>{0, 1});
    
    // Domain for G
    rcptr< vector<T> > dom3(new vector<T>{0, 1, 2});

    //*********************************************************
    // Define the RVs
    //*********************************************************

    // The enum statement here predefines two RV ids: the id of X is 0
    // and the id of Y is 1. This is easy enough in very simple
    // problems, for more complex situations involving many RVs this
    // becomes cumbersome and we will need a datastructure such as a
    // map to save the RV ids in. Consult the userguide for more on
    // this.

    enum { D, I, S, G, L };



    // define factors
    rcptr<Factor> pD = uniqptr<DT>(new DT(
      {D}, {dom2}, defProb, {
        {{0}, 0.6},
        {{1}, 0.4}
    }));
    std::cout << "--- p(D) ---\n" << *pD << std::endl;

    rcptr<Factor> pI = uniqptr<DT>(new DT(
      {I}, {dom2}, defProb, {
        {{0}, 0.7},
        {{1}, 0.3}
    }));
    std::cout << "--- p(I) ---\n" << *pI << std::endl;

    rcptr<Factor> pS_I = uniqptr<DT>(new DT(
      {I, S}, {dom2, dom2}, defProb, {
        {{0, 0}, 0.95}, {{0, 1}, 0.05},
        {{1, 0}, 0.20}, {{1, 1}, 0.80}
    }));
    std::cout << "--- p(S | I) ---\n" << *pS_I << std::endl;


    rcptr<Factor> pG_ID = uniqptr<DT>(new DT(
      {I, D, G}, {dom2, dom2, dom3}, defProb, {
        {{0, 0, 0}, 0.30}, {{0, 0, 1}, 0.40}, {{0, 0, 2}, 0.30},
        {{0, 1, 0}, 0.05}, {{0, 1, 1}, 0.25}, {{0, 1, 2}, 0.70},
        {{1, 0, 0}, 0.90}, {{1, 0, 1}, 0.08}, {{1, 0, 2}, 0.02},
        {{1, 1, 0}, 0.50}, {{1, 1, 1}, 0.30}, {{1, 1, 2}, 0.20}
    }));
    std::cout << "--- p(G | I, D) ---\n" << *pG_ID << std::endl;

    rcptr<Factor> pL_G = uniqptr<DT>(new DT(
      {G, L}, {dom3, dom2}, defProb, {
        {{0, 0}, 0.10}, {{0, 1}, 0.90},
        {{1, 0}, 0.40}, {{1, 1}, 0.60},
        {{2, 0}, 0.99}, {{2, 1}, 0.01}
    }));
    std::cout << "--- p(L | G) ---\n" << *pL_G << std::endl;

    //2.1 a
    rcptr<Factor> joint = pD->absorb(pI)->absorb(pS_I)->absorb(pG_ID)->absorb(pL_G);
    std::cout << "--- Joint Distribution P(D, I, S, G, L) ---\n" << *joint << std::endl;

    //2.1 b
    rcptr<Factor> margL_given_DS = joint->observeAndReduce({D, S}, {1, 1})->marginalize({L})->normalize();
    std::cout << "--- P(L | D=1, S=1) ---\n" << *margL_given_DS << std::endl;

    //2.2 a
    // if I is unobserved, sat score and grade is dependent, but if I is observed, they are independent

    //2.2 b
    // Marginalize G before observing
    rcptr<Factor> margG = joint->marginalize({G})->normalize();
    std::cout << "--- P(G) ---\n" << *margG << std::endl;

    // Observe S = 1, I unobserved
    rcptr<Factor> margG_given_S = joint->observeAndReduce({S}, {1})->marginalize({G})->normalize();
    std::cout << "--- P(G | S=1) ---\n" << *margG_given_S << std::endl;

    // notice this changes cause S and G are dependent

    // Observe I = 1, S unobserved
    rcptr<Factor> margG_given_I = joint->observeAndReduce({I}, {1})->marginalize({G})->normalize();
    std::cout << "--- P(G | I=1) ---\n" << *margG_given_I << std::endl;

    // Observe I = 1, S = 1
    rcptr<Factor> margG_given_IS = joint->observeAndReduce({I, S}, {1, 1})->marginalize({G})->normalize();
    std::cout << "--- P(G | I=1, S=1) ---\n" << *margG_given_IS << std::endl;

    //notice this does not change cause G is conditionally independent of S given I

  
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
