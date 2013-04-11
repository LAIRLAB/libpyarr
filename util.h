/*
  Author: Nick Rhinehart
 * National Robotics Engineering Center, Carnegie Mellon University
 * Copyright © 2012, Confidential and Proprietary, All rights reserved.
 */

#ifndef _NICKS_UTIL_H
#define _NICKS_UTIL_H

#include <iostream>
#include <string>
#include <exception>
#include <fstream>
#include <vector>

#define UTIL_DEBUG 0

using std::string;
using std::endl;
using std::ifstream;
using std::cerr;



namespace file_util{

  //variations on function to require existence of a file or 
  //exit with an error.
  void require_existence(string path) {
    ifstream test(path.c_str());
    if(!test) {
      stringstream ss; ss<<"path: '"<<path<<"' doesn't exist";
      //print_with_error(ss.str().c_str());
    }
    else{test.close();}
  }
  void require_existence(string path,string err) {
    ifstream test(path.c_str());
    if(!test) {
      stringstream ss; ss<< err << " path: '"<<path<<"' doesn't exist.";
      //print_with_error(ss.str().c_str());
    }
    else{test.close();}
  }
   void require_existence(string path,const char *e) {
    ifstream test(path.c_str());
    string err(e);
    if(!test) {
      stringstream ss; ss<< err << " path: '"<<path<<"' doesn't exist.";
      //print_with_error(ss.str().c_str());
    }
    else{test.close();}
  }
   
   static vector<string> load_lines(string path) {
     ifstream in(path.c_str());
     if(in.is_open() == false) {
       throw std::invalid_argument("Could not load lines from path: "+path);
     }
     string line;
     vector<string> vec_strings;
     while(getline(in,line).eof() == false) {
       size_t carriage_pos = line.find_last_not_of("\r\n");
       if(carriage_pos != string::npos) {
	 line.erase(carriage_pos + 1);
       }
       vec_strings.push_back(line);
     }
     in.close();
     return vec_strings;
   }
}

#endif
