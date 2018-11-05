# f4mul
F_4 (4th Fermat number) arithmetic on 16 bits

This simulator performs arithmetic on the F_4 field (F_2^2^4+1 = 65537).

It currently performs the vector addition, substraction and Hadamard multiplication (entrywise product) of numbers in this field modulo 65537 on 16 bits. The modular multiplication is drastically simplified and only involves 2 multiplications for (1-16/65537) of the cases in dimension 16.

This is useful for computing the NTT (Fast Fourier Transform) in the F_4 field.
