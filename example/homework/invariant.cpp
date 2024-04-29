// these have to be filled in correctly: PxQ grid
int P, Q, M;

// process (p,q):
int p, q, m, n;

// x is replicated in each process column, distributed vertically
// y is replicated in each process row   , distributed horizontally
// global lengths of both vectors is M.
// m -- local length of x[] in process (p,q)
// n -- local length of y[] in process (p,q)
//

A) Linear x, Linear y

int nominal1 = M/P; int extra1 = M%P;
int nominal2 = M/Q; int extra2 = M%Q;

for(i = 0; i < m; i++) // m is the local size of the vector x[]
{ 
    // x local to global: given that this element is (p,i), what is its global index I?
  int I = i + ((p < extra1) ? (nominal1+1)*p :
	       (extra1*(nominal1+1)+(p-extra1)*nominal1));

   // so to what (qhat,jhat) does this element of the original global vector go?
   int qhat = (I < extra2*(nominal2+1)) ? I/(nominal2+1) : 
                                          (extra2+(I-extra2*(nominal2+1))/nominal2);
   int jhat = I - ((qhat < extra2) ? (nominal2+1)*qhat :
		   (extra2*(nominal2+1) + (qhat-extra2)*nominal2));

   if(qhat == q)  // great, this process has an element of y!
   { 
      y_local[jhat] = x_local[i];
   }
}

B) Linear x, Scatter y

int nominal1 = M/P; int extra1 = M%P;

for(i = 0; i < m; i++) // m is the local size of the vector x[]
{ 
    // x local to global: given that this element is (p,i), what is its global index I?
  int I = i + ((p < extra1) ? (nominal1+1)*p :
	       (extra1*(nominal1+1)+(p-extra1)*nominal1));

   // so to what (qhat,jhat) does this element of the original global vector go?
   int qhat = I%Q;
   int jhat = I/Q;

   if(qhat == q)  // great, this process has an element of y!
   { 
      y_local[jhat] = x_local[i];
   }
}

C) Scatter x, Scatter y

for(i = 0; i < m; i++) // m is the local size of the vector x[]
{ 
    // x local to global: given that this element is (p,i), what is its global index I?
   int I = i*P + p;

   // so to what (qhat,jhat) does this element of the original global vector go?
   int qhat = I%Q;
   int jhat = I/Q;

   if(qhat == q)  // great, this process has an element of y!
   { 
      y_local[jhat] = x_local[i];
   }
}

D) Scatter x, Linear y

int nominal2 = M/Q; int extra2 = M%Q;

for(i = 0; i < m; i++) // m is the local size of the vector x[]
{ 
    // x local to global: given that this element is (p,i), what is its global index I?
   int I = i*P + p;

   // so to what (qhat,jhat) does this element of the original global vector go?
   int qhat = (I < extra2*(nominal2+1)) ? I/(nominal2+1) : 
                                    (extra2+(I-extra2*(nominal2+1))/nominal2);
   int jhat = I - ((qhat < extra2) ? (nominal2+1)*qhat :
		   (extra2*(nominal2+1) + (qhat-extra2)*nominal2));

   if(qhat == q)  // great, this process has an element of y!
   { 
      y_local[jhat] = x_local[i];
   }
}

