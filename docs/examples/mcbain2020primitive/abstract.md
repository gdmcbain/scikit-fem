# The primitive Orr-Sommerfeld equation and its solution by finite elements

The linear stability of one-dimensional viscous shear flows is classically governed by the Orr–Sommerfeld equation in the disturbance streamfunction.  This fourth-order equation is obtained from the second-order Navier–Stokes equation by eliminating the pressure.

Here we consider retaining the primitive velocity–pressure formulation as is required for the linear stability analysis in general multidimensional geometries for which the streamfunction is unavailable; this conveniently reduces the conceptual and notational step from one to higher dimensions.

Further, multidimensional flow simulation in arbitrary geometry is generally based on the primitive Navier–Stokes equations; having the same formulation and discretization in the linear stability analysis simplifies the comparison, removing possible numerical causes of discrepancy.

The Orr-Sommerfeld equation is here discretized using Python and scikit-fem, in classical and primitive forms with Hermite-beam and Taylor–Hood elements respectively.  The solutions for the standard test problem of plane Poiseuille flow show the primitive formulation to be simple, clear, accurate, and economic.
