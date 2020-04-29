using CuArrays
using LinearAlgebra
using IterativeSolvers

#This file contains the incomplete factorization routine

#This function computes the inner procuct of two sparse vectors that are 
#stores using the same nzind and nzval, but different offsets iter1:u1 resp.
#iter2:u2
function _innerprod( iter1::Int64,
                     u1::Int64,
                     iter2::Int64,
                     u2::Int64,
                     nzind::Array{Int64,1},
                     nzval::Array{Float64,1} )
  @inbounds begin
    res = zero(Float64)
    while ( iter1 <= u1 ) && ( iter2 <= u2 )
      while (nzind[iter1] == nzind[iter2])  &&( iter1 <= u1 ) && ( iter2 <= u2 )
        res += nzval[iter1] * nzval[iter2]
        iter1 += 1
        iter2 += 1
      end
      if nzind[iter1] < nzind[iter2]
        iter1 += 1
      else
        iter2 += 1
      end
    end
    return res
  end
end

function _innerprod( range1,
                     range2,
                     nzind::Array{Int64,1},
                     nzval::Array{Float64,1} )
  iter1 = range1.start
  u1 = range1.stop

  iter2 = range2.start
  u2 = range2.stop
  @inbounds begin
    res = zero(Float64)
    while ( iter1 <= u1 ) && ( iter2 <= u2 )
      while (nzind[iter1] == nzind[iter2])  &&( iter1 <= u1 ) && ( iter2 <= u2 )
        res += nzval[iter1] * nzval[iter2]
        iter1 += 1
        iter2 += 1
      end
      if nzind[iter1] < nzind[iter2]
        iter1 += 1
      else
        iter2 += 1
      end
    end
    return res
  end
end



##This function computes the in place incomplete Cholesky factorisation using 
##an inner product wrapped in a function.
#function _el_icholRWHL!( s::Array{Float64},
#                          ind::Array{Int64, 1}, 
#                          jnd::Array{Int64, 1}, 
#                          )
#  @inbounds begin
#    #piv_i goes from 1 to N and designates the number, which is being treated 
#    for piv_i = 1 : ( size( ind, 1 ) - 1 ) 
#      #piv_j ranges over pointers and designates where to find the present column
#      for piv_j = ind[ piv_i ] : ( ind[ piv_i + 1 ] - 1 )
#        #iter designates the pointer, at which the i-iterator starts
#        iter::Int64 = ind[ piv_i ]
#        #jter designates the pointer, at which the j-iterator starts
#        jter::Int64 = ind[ jnd[ piv_j ] ]
#        #The condition makes sure, that only the columns up to th pivot 
#        #are being summed.
#  
#        s[ piv_j ] -= _innerprod( iter, ind[ piv_i + 1 ] - 2,
#                                  jter, ind[ jnd[ piv_j ] + 1 ] - 2,
#                                  jnd,
#                                  s)
#   
#        if jnd[ piv_j ] < piv_i
#          s[ piv_j ] /= ( s[ ind[ jnd[ piv_j ] + 1 ] - 1 ] )
#        elseif jnd[ piv_j ] == piv_i
#          if ( s[ ind[ piv_i + 1 ] - 1 ] ) <= 0.
#            s[ ind[ piv_i + 1 ] - 1 ] = 0
#            #Treating the case where the matrix is near-low rank and our 
#            #factorization returns a low-rank matrix.
#            maxj = piv_i
#            for newpiv_i = ( piv_i + 1 ) : ( size( ind, 1 ) - 1 )               
#              newpiv_j::Int64 = ind[ newpiv_i ] 
#              while jnd[ newpiv_j ] < maxj
#                #iter designates the pointer, at which the i-iterator starts
#                iter = ind[ newpiv_i ]
#                #jter designates the pointer, at which the j-iterator starts
#                jter = ind[ jnd[ newpiv_j ] ]
#                #The condition makes sure, that only the columns up to th pivot 
#                #are being summed.
#          
#                s[ newpiv_j ] -= _innerprod( iter, ind[ newpiv_i + 1 ] - 2,
#                                          jter, ind[ jnd[ newpiv_j ] + 1 ] - 2,
#                                          jnd,
#                                          s)
#
#                s[ newpiv_j ] /= ( s[ ind[ jnd[ newpiv_j ] + 1 ] - 1 ] )
#                newpiv_j += 1
#              end
#              while newpiv_j <= ( ind[ newpiv_i + 1 ] - 1 ) 
#                s[ newpiv_j ] = 0.
#                newpiv_j += 1
#              end
#            end
#            println( "Pivot in i = ", piv_i, " is nonpositive", 
#                     " returning low rank approximation" )
#            return 0
#          else
#            s[ piv_j ] = sqrt( s[ piv_j ] ) 
#          end
#        else
#          println( "Warning: jnd[ piv_j ] > piv_i" )
#        end
#      end
#    end
#  end
#end

#This function computes the in place incomplete Cholesky factorisation using 
#an inner product wrapped in a function.
function _el_icholRWHL!( s::Array{Float64},
                          ind::Array{Int64, 1}, 
                          jnd::Array{Int64, 1}, 
                          )
  @inbounds begin
    #piv_i goes from 1 to N and designates the number, which is being treated 
    for piv_i = 1 : ( size( ind, 1 ) - 1 ) 
      #piv_j ranges over pointers and designates where to find the present column
      for piv_j = ind[ piv_i ] : ( ind[ piv_i + 1 ] - 1 )
        #iter designates the pointer, at which the i-iterator starts
        iter::Int64 = ind[ piv_i ]
        #jter designates the pointer, at which the j-iterator starts
        jter::Int64 = ind[ jnd[ piv_j ] ]
        #The condition makes sure, that only the columns up to th pivot 
        #are being summed.
  
        s[ piv_j ] -= _innerprod( iter, ind[ piv_i + 1 ] - 2,
                                  jter, ind[ jnd[ piv_j ] + 1 ] - 2,
                                  jnd,
                                  s)
   
        if ( s[ ind[ jnd[ piv_j ] + 1 ] - 1 ] ) > 0.0
          if jnd[ piv_j ] < piv_i
            s[ piv_j ] /= ( s[ ind[ jnd[ piv_j ] + 1 ] - 1 ] )
            #debugging:
            if isnan( s[ piv_j ] ) || isinf( s[ piv_j ] )
              println( "ALERT! Not a number" )
            end
            #end debugging

          else 
            #debugging:
            if ind[ piv_i + 1 ] - 1 != piv_j
              println( "ALERT! Not selecting diagonal element" )
            end
            #end debugging
            s[ piv_j ] = sqrt( s[ piv_j ] ) 
            #debugging:
            if isnan( s[ piv_j ] ) || isinf( s[ piv_j ] )
              println( "ALERT! Not a number" )
            end
            #end debugging

          end
        else
          s[ piv_j ] = 0.0
        end
      end
    end
  end
end


#In-place Cholesky factorization of an uppper triangular matrix
function icholU_high_level!( U::SparseMatrixCSC{Float64, Int64} )
  @fastmath _el_icholRWHL!( U.nzval, U.colptr, U.rowval )
end

function icholGPU!( U::SparseMatrixCSC{Float64,Int64} )
  #prevent loss of structural zeros:
  U.nzval .+= eps(Float32)
  d_U = CuArrays.CUSPARSE.CuSparseMatrixCSC( U ) 
  CuArrays.CUSPARSE.ic02!(d_U, 'O' );

  U.nzval .= Array{Float64}( d_U.nzVal )
end

# probably not needed 
# function UfromLTL(σ, L)
#     U = sparse(L')
#     for j = 1 : size(U,2)
#       for ind = U.colptr[j] : (U.colptr[j+1] - 1)
#         i = U.rowval[ind]
#         U.nzval[ind] = L[:,i]' *invσ * L[:,j]
#         if i == j
#           U.nzval[ind] += 1
#         end
#       end
#     end
#     return U
# end

struct IChol
  L
  U
  # optional additive diagonal
invσ
  tmp
end

function IChol(L, U)
  return IChol(L, U, zeros(eltype(L), size(L,1)), Vector{eltype(L)}(undef, size(L,1)))
end

function IChol(L, U, invσ)
  return IChol(L, U, invσ, Vector{eltype(L)}(undef, size(L,1)))
end


import Base.\
import Base.*
import LinearAlgebra.ldiv!
import LinearAlgebra.lmul!
import LinearAlgebra.mul!
import LinearAlgebra.size 
import LinearAlgebra.eltype 
# Since ldiv! presently does not support sparse triangular matrices, 
# the optimized non-allocating versions have been replaced with allocating 
# versions (which dispatch to BLAS calls).

function size(A::IChol)
  return size(A.L)
end

function size(A::IChol, dim)
  return size(A.L, dim)
end


function eltype(A::IChol)
  return eltype(A.L)
end

function ldiv!(y, A::IChol, x)
  # ldiv!(A.tmp, A.L, x) 
  # ldiv!(y, A.U, A.tmp) 
  y .= A.U \ (A.L \ x)
end

function ldiv!(A::IChol, x)
  # ldiv!(A.tmp, A.L, x) 
  # ldiv!(x, A.U, A.tmp) 
  x .= A.U \ (A.L \ x)
end

function \(A::IChol, x)
  # ldiv!(A.tmp, A.L, x) 
  # return A.U \ A.tmp
  return A.U \ (A.L \ x)
end

function *(A::IChol, x)
  # y = A.U * x 
  # lmul!(A.L, y)
  # y .+= A.invσ .* x 
  #return y
  return A.L * (A.U * x) .+ A.invσ .* x
end

function mul!(y, A::IChol, x)
  # y .= x
  # lmul!(A.U, y) 
  # lmul!(A.L, y) 
  # y .+= A.invσ .* x
  y .= A.L * (A.U * x) .+ A.invσ .* x
end


function lmul!(A::IChol, x)
  # A.tmp .= A.invσ .* x
  # lmul!(A.U, x) 
  # lmul!(A.L, x) 
  # x .+= A.tmp
  y = A.L * (A.U * x)
  y = y + A.invσ .* x
  x .= y
end



# in place squaring of a lower triangular matrix to an upper triangular matrix

function squareSparse(L)
  UTemp = sparse(L')
  U = similar(UTemp)
  U.nzval .= UTemp.nzval
  for i = 1 : U.n
    for j in nzrange(U, i)
      rowj = U.rowval[j]
      U.nzval[j] = _innerprod(nzrange(UTemp,i), nzrange(UTemp,rowj), UTemp.rowval, UTemp.nzval)
    end
  end
  return U
end

struct NoiseCov
  L
  U
  invσ

  LICholσ
  LIChol
  UIChol

  function NoiseCov(L, U,σ)
    LICholσ = IChol(L, L', 1 ./ σ)
    LIChol = IChol(L, L')
    UIChol = IChol(U', U)
    return new(L, U, 1 ./ σ, LICholσ, LIChol, UIChol)
  end
end

function ldiv!(y, A::NoiseCov, x)
  y .= A.invσ .* cg(A.LICholσ, A.LIChol * x, Pl=A.UIChol, maxiter=5, tol=1e-16, verbose=true)
end

function ldiv!(A::NoiseCov, x)
  x .= A.invσ .* cg(A.LICholσ, A.LIChol * x, Pl=A.UIChol, maxiter=5, tol=1e-16, verbose=true)
end

function \(A::NoiseCov, x)
  return A.invσ .* cg(A.LICholσ, A.LIChol * x, Pl=A.UIChol, maxiter=5, tol=1e-16, verbose=true)
end

function *(A::NoiseCov, x)
  return A.LIChol \ (A.LICholσ *( A.invσ .\ x))
end

function lmul!(y, A::NoiseCov, x)
  lmul!(y, A.LICholσ, A.invσ .\ x)
  y .*= A.LIChol \ y
end

function lmul!(A::NoiseCov, x)
  lmul!(A.LICholσ, A.invσ .\ x)
  x .= A.LIChol \ x
end

import LinearAlgebra.logdet
function logdet(A::NoiseCov)
  return 2 * logdet(UpperTriangular(A.U)) - 2 * logdet(UpperTriangular(A.L)) + sum(log.(A.invσ))
end