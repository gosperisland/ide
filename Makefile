#TODO: consider auto-generating with eclipse

all:	LearningMain


MAINNAME= LearningMain

#########################################################
# compiler and its flags 
#########################################################
CXX= g++

CXXFLAGS= -Wall -Wvla -Werror -g -pg -std=c++11  

EXTERNAL_LIBS= -larmadillo 
#########################################################

##########################################################
# sources files
##########################################################


HEADERFILES= Learning.hpp ID/IDpair.hpp ID/ID.hpp ID/Grid.hpp ID/Pair.hpp 


OLDVERSIONHEADERS= ID01/IDpair.hpp ID01/ID.hpp ID01/Grid.hpp ID01/Pair.hpp
##########################################################

clean:
	rm *.o -f ${MAINNAME}
	

${MAINNAME}: ${MAINNAME}.o	
	${CXX} ${CXXFLAGS}  ${MAINNAME}.o -o $@ ${EXTERNAL_LIBS}

${MAINNAME}.o:  ${MAINNAME}.cpp ${HEADERFILES} 
	${CXX} ${CXXFLAGS} -c  ${MAINNAME}.cpp
	

