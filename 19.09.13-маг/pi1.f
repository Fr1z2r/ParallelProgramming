c**********************************************************************
c   pi.f - compute pi by integrating f(x) = 4/(1 + x**2)     
c     
c   Each node: 
c    1) receives the number of rectangles used in the approximation.
c    2) calculates the areas of it's rectangles.
c    3) Synchronizes for a global summation.
c   Node 0 prints the result.
c
c  Variables:
c
c    pi  the calculated result
c    n   number of points of integration  
c    x           midpoint of each rectangle's interval
c    f           function to integrate
c    sum,pi      area of rectangles
c    tmp         temporary scratch space for global summation
c    i           do loop index
c****************************************************************************
      program main

      include 'mpif.h'

      double precision  PI25DT
      parameter        (PI25DT = 3.141592653589793238462643d0)

      double precision  pi, h, sum, x
      double precision  t1,t2

      OPEN (2,FILE='result')

c     function to integrate

      call MPI_INIT( ierr )

      t1 = MPI_WTime()

c  number of points of integration  

      n=10000000

c  calculate the interval size

      h = 1.0d0/n

      sum  = 0.0d0
      do 20 i = 1, n
         x = h * (i - 0.5d0)
         x= 4.d0 / (1.d0 + x*x)
         sum = sum + x
 20   continue
      pi = h * sum

      t2 = MPI_WTime()

c     node 0 prints the answer.
         write(*, 97) pi, abs(pi - PI25DT)
 97      format('  pi is approximately: ', F18.16,
     #          '  Error is: ', F18.16)
         t=t2-t1
         write(*,*)'time =',t,'sec'

       WRITE(2,3) t
 3     FORMAT(5X,'time =',F10.5,1X,'SEC')

      call MPI_FINALIZE( ierr )
      stop
      end




