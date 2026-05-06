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
    rcptr< vector<T> > doorDom(new vector<T>{0, 1, 2}); // Domain for the 3 doors

    //*********************************************************
    // Define the RVs
    //*********************************************************

    // The enum statement here predefines two RV ids: the id of X is 0
    // and the id of Y is 1. This is easy enough in very simple
    // problems, for more complex situations involving many RVs this
    // becomes cumbersome and we will need a datastructure such as a
    // map to save the RV ids in. Consult the userguide for more on
    // this.

    enum{I, C, M, R};



    // define factors
    rcptr<Factor> pI = 
      uniqptr<DT>(
        new DT(
          {I},
          {doorDom},
          defProb,
          {
            {{0}, 1.0/3.0},
            {{1}, 1.0/3.0},
            {{2}, 1.0/3.0}
      } ));

    std::cout << __FILE__ << __LINE__ << ": " << *pI << std::endl;

    // define the factor P(C)
    rcptr<Factor> pC = 
      uniqptr<DT>(
        new DT(
          {C},
          {doorDom},
          defProb,
          {
            {{0}, 1.0/3.0},
            {{1}, 1.0/3.0},
            {{2}, 1.0/3.0}
      } ));

    std::cout << __FILE__ << __LINE__ << ": " << *pC << std::endl;

    // p(M | I, C)
    rcptr<Factor> pM_IC = uniqptr<DT>(new DT(
      {I, C, M}, {doorDom, doorDom, doorDom}, defProb, {
        {{0, 0, 1}, 0.5}, 
        {{0, 0, 2}, 0.5},
        {{0, 1, 2}, 1.0},
        {{0, 2, 1}, 1.0},
        {{1, 0, 2}, 1.0},
        {{1, 1, 0}, 0.5}, 
        {{1, 1, 2}, 0.5},
        {{1, 2, 0}, 1.0},
        {{2, 0, 1}, 1.0},
        {{2, 1, 0}, 1.0},
        {{2, 2, 0}, 0.5}, 
        {{2, 2, 1}, 0.5}
    }));

    std::cout << __FILE__ << __LINE__ << ": " << *pM_IC << std::endl;


  // (1.1 a) Joint distribution
  rcptr<Factor> jointICM = pI->absorb(pC)->absorb(pM_IC);
  std::cout << __FILE__ << __LINE__ << ": P(I,C,M) = " << *jointICM << std::endl;

  // (1.1 b) Cars location
  rcptr<Factor> C_IM = jointICM->observeAndReduce({I, M}, {0, 1})->marginalize({C})->normalize();
  std::cout << ":p(C | I=0, M=1)" << *C_IM << std::endl;

  //1.2 a
  // If M is unobserved:
  // tells you nothing. The car is placed randomly and your initial choice doesn't move the car. I and C are independent.

  // If M is observed?
  // If you know which door you chose (I), and you observe which door Monty opens (M), you get info about where the car is. For example, if you chose 0, and Monty opens 1, you know the car cannot be behind 1, and the probability of it being behind 2 increases to 66.7%. Observing M opens the path making I and C dependent.

  //1.2 b
  // Marginalize C
  rcptr<Factor> margC = jointICM->marginalize({C})->normalize();
  std::cout << "P(C)" << *margC << std::endl;

  // Observe I = 0,  M unobserved
  rcptr<Factor> margC_given_I = jointICM->observeAndReduce({I}, {0})->marginalize({C})->normalize();
  std::cout << "P(C | I=0)" << *margC_given_I << std::endl;

  // Observe M = 1, I unobserved
  rcptr<Factor> margC_given_M = jointICM->observeAndReduce({M}, {1})->marginalize({C})->normalize();
  std::cout << "P(C | M=1)" << *margC_given_M << std::endl;

  // Observe M = 1, set I = 0
  rcptr<Factor> margC_given_MI = jointICM->observeAndReduce({M, I}, {1, 0})->marginalize({C})->normalize();
  std::cout << "P(C | M=1, I=0)" << *margC_given_MI << std::endl;

  // 1.2 c
  rcptr<Factor> pR_M = uniqptr<DT>(new DT(
      {M, R}, {doorDom, doorDom}, defProb, {
        {{0, 0}, 0.8}, 
        {{0, 1}, 0.1}, 
        {{0, 2}, 0.1},
        {{1, 0}, 0.1}, 
        {{1, 1}, 0.8}, 
        {{1, 2}, 0.1},
        {{2, 0}, 0.1}, 
        {{2, 1}, 0.1}, 
        {{2, 2}, 0.8}
    }));
    std::cout << "p(R | M)\n" << *pR_M << std::endl;

    rcptr<Factor> jointICMR = jointICM->absorb(pR_M);
    
    std::cout << "Joint P(I, C, M, R)\n" << *jointICMR << std::endl;

    //1.2 d
    //not observing R tells you nothing about the car's location
    // observing R gives you info about M, which in turn gives you info about C

    // 1.2 e
    // Marginalalize C before observing
    rcptr<Factor> margC_2 = jointICMR->marginalize({C})->normalize();
    std::cout << "P(C) Prior\n" << *margC_2 << std::endl;

    // Observe I = 0, R unobserved
    rcptr<Factor> margC_given_I2 = jointICMR->observeAndReduce({I}, {0})->marginalize({C})->normalize();
    std::cout << "P(C | I=0)\n" << *margC_given_I2 << std::endl;

    // Observe R = 1, I unobserved
    rcptr<Factor> margC_given_R = jointICMR->observeAndReduce({R}, {1})->marginalize({C})->normalize();
    std::cout << "P(C | R=1)\n" << *margC_given_R << std::endl;

    // Observe R = 1 set I = 0
    rcptr<Factor> margC_given_RI = jointICMR->observeAndReduce({R, I}, {1, 0})->marginalize({C})->normalize();
    std::cout << "P(C | R=1, I=0)\n" << *margC_given_RI << std::endl;

    // 1.2 f
    //No we cannot reverse the direction of the arrows. Doing so would destroy our current v structure which would fundamentally change the rules of the game. Flipping for example arrow from m to r causes 2 new v structures to be formed. 

    // 1.2 g
    // Observe R = 0, I = 0
    rcptr<Factor> margC_given_RI_same = jointICMR->observeAndReduce({R, I}, {0, 0})->marginalize({C})->normalize();
    std::cout << "P(C | R=0, I=0)\n" << *margC_given_RI_same << std::endl;
    // we know the reporter is lying this doesnt add any mathematical info about the location. so the probabilities go back to being 1/3 for each door

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
