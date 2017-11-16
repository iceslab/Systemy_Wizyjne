CXXFLAGS=-c -Wall -std=c++14
LDFLAGS=-lopencv_imgcodecs -lopencv_features2d -lopencv_xfeatures2d -lopencv_highgui -lopencv_core
SOURCES=algorithms.cpp surf.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=surf

all: release

release: CXXFLAGS += -O3
release: $(SOURCES) $(EXECUTABLE)
	
debug: CXXFLAGS += -gfull
debug: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -o $@