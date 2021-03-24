# ------------------------------------------------------------------------------
#  Compile with C++ 17
# ------------------------------------------------------------------------------
SRCS=pri_queue.cc util.cc qalsh.cc rqalsh.cc baseline.cc eh.cc bh.cc mh.cc \
	orig_eh.cc orig_bh.cc orig_mh.cc fh.cc fh_wo_s.cc fh_minus.cc \
	fh_minus_wo_s.cc nh.cc nh_wo_s.cc nh_counting.cc nh_counting_wo_s.cc \
	ap2h.cc main.cc
OBJS=${SRCS:.cc=.o}

CXX=g++-8 -std=c++17
CPPFLAGS=-w -O3

.PHONY: clean

all: ${OBJS}
	${CXX} ${CPPFLAGS} -o p2h ${OBJS}

clean:
	-rm ${OBJS} p2h
