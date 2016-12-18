all:	LearningMain


MAINNAME= LearningMain

#########################################################
# compiler and its flags 
#########################################################
CXX= g++

CXXFLAGS= -Wall -Wvla -Werror -g -std=c++11  


CXXLINKFLAGSTEST= -larmadillo 
#########################################################

##########################################################
# sources files
##########################################################


HEADERFILES= Learning.hpp ID/IDpair.hpp ID/ID.hpp ID/Grid.hpp ID/Pair.hpp 


OLDVERSIONHEADERS= ID01/IDpair.hpp ID01/ID.hpp ID01/Grid.hpp ID01/Pair.hpp
##########################################################

clean:
	rm *.o -f
	
main:
	g++ -Wall -Wvla -Werror -g -D_GLIBCXX_DEBUG -std=c++11 main.cpp -o example
	#./example
	
	#g++ -Wall -g -std=c++11  -c test_Learning.cpp  
	#g++ -Wall -g -std=c++11  test_Learning.o -o test-learning
${MAINNAME}: ${MAINNAME}.o	
	g++ -Wall -g -pg -std=c++11  ${MAINNAME}.o -o LearningMain -larmadillo 

${MAINNAME}.o:  ${MAINNAME}.cpp ${HEADERFILES} 
	g++  -Wall -g -pg -std=c++11  -c  ${MAINNAME}.cpp
	

