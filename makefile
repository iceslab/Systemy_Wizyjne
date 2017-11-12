CFLAGS=-c -Wall
LDFLAGS=-v -lopencv_imgcodecs -lopencv_features2d -lopencv_xfeatures2d -lopencv_highgui -lopencv_core
SOURCES=surf.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=surf

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(LDFLAGS) $(LIB) $(OBJECTS) -o $@

.cpp.o:
	$(CXX) $(CFLAGS) $(INC) $< -o $@