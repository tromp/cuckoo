// parameter map for solvers to faciliate linked code
#include <unordered_map>
using namespace std;

#ifndef CALL_CONVENTION
#define CALL_CONVENTION
#endif

class ParamMap {
	public:
		void set(string name, int value);
		int get(string name);
		// call above in a more linker-friendly way
		CALL_CONVENTION void set_param(const char* name, unsigned int name_len, int value);
		CALL_CONVENTION int get_param(const char* name, unsigned int name_len);
	private:
		unordered_map<string, int> pmap;
};

// set a parameter for the solver
void ParamMap::set(string name, int value){
	pmap[name] = value;
}

// get a parameter from the solver
int ParamMap::get(string name){
	return pmap[name];
}

// intended for linkage
CALL_CONVENTION void ParamMap::set_param(const char* name, unsigned int name_len, int value) {
	set(string(name, name_len), value);
}

// intended for linkage
CALL_CONVENTION int ParamMap::get_param(const char* name, unsigned int name_len){
	return get(string(name, name_len));
}
