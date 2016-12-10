all:	LearningDebug
	
main:
	g++ -Wall -Wvla -Werror -g -D_GLIBCXX_DEBUG -std=c++11 main.cpp -o example
	#./example
	
	#g++ -Wall -g -std=c++11  -c test_Learning.cpp  
	#g++ -Wall -g -std=c++11  test_Learning.o -o test-learning
LearningDebug: LearningDebug.o	
	g++ -Wall -g -pg -std=c++11  LearningDebug.o -o LearningDebug -larmadillo 

LearningDebug.o: LearningDebug.cpp  IDpair.hpp ID.hpp Grid.hpp Pair.hpp
	g++  -Wall -g -pg -std=c++11  -c  LearningDebug.cpp
	

