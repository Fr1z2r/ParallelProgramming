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
c    n   number of points of integration.  
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

      double precision  mypi, pi, h, sum, x, f, a
      double precision  t1,t2
      integer n, myid, numprocs, i, rc

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
      print *, 'Process ', myid, ' of ', numprocs, ' is alive'
      
      t1 = MPI_WTime()

      if ( myid .eq. 0 ) then
       n=1000000000
      endif
      
      call MPI_BCAST(n,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)


c     calculate the interval size
      h = 1.0d0/n

      sum  = 0.0d0

      do 20 i = myid+1, n, numprocs
         x = h * (i - 0.5d0)
         x= 4.d0 / (1.d0 + x*x)
         sum = sum + x
 20   continue
      mypi = h * sum

c     collect all the partial sums

      call MPI_REDUCE(mypi,pi,1,MPI_DOUBLE_PRECISION,MPI_SUM,0,
     $     MPI_COMM_WORLD,ierr)

      t2 = MPI_WTime()
c                                 node 0 prints the answer.
      if (myid .eq. 0) then
         write(*, 97) pi, abs(pi - PI25DT)
 97      format('  pi is approximately: ', F18.16,
     +          '  Error is: ', F18.16)

         t=t2-t1
         write(*,*)'time =',t,'sec'

      endif


 30   call MPI_FINALIZE(rc)
      stop
      end




