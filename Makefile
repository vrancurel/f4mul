
CFLAGS = -Wunused -mavx2 -g #-O3

OBJS = f4mul.o

f4mul: $(OBJS)
	cc $(OBJS) -o f4mul

clean:
	rm -f $(OBJS) f4mul
