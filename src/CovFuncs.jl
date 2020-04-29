#This file implements a number of popular covariance functions
using SpecialFunctions
using IntelVectorMath
function matern( r::Float64, l::Float64, nu::Float64 )
  if r == 0
    return 1.0
  else
    return 2.0^(1.0 - nu) / gamma(nu) * (sqrt( 2.0 * nu ) * r / l )^nu * besselk( nu, sqrt( 2.0 * nu ) * r / l ) 
  end
end

function matern12( r, l )
  exp( - r / l )
end

function matern12!( r, l )
  r ./= -l
  IntelVectorMath.exp!(r)
end


function matern32( r, l )
  ( 1 + sqrt(3) * r / l ) * exp( - sqrt(3) * r / l )
end

function matern32!( r, l )
  r .= - sqrt(3) .* r ./ l
  rtemp = 1 .- r
  IntelVectorMath.exp!(r)
  r .*= rtemp
end


function matern52( r, l )
  ( 1 + sqrt(5) * r / l + 5 * r^2 / 3 / l^2 ) * exp( - sqrt(5) * r / l )
end

function matern52!( r, l )
  rtemp = ( 1 .+ sqrt(5) * r / l .+ 5 * r.^2 / 3 / l^2 )
  r .= -sqrt(5) .* r ./ l
  IntelVectorMath.exp!(r)
  r .*= rtemp
end


function gaussian( r::Float64, l::Float64 )
  return exp( - ( r/ l )^2 )
end

function exponential( r::Float64, l::Float64 )
  return exp( -r/l )
end

function exponential!(r, l)
  r ./= -l
  IntelVectorMath.exp!(r)
end


function invMultiquadratic( r::Float64, l::Float64, c::Float64 = 1. )
  return 1/sqrt( c + ( r / l )^2 )
end

function ratQuadratic( r::Float64, l::Float64, c::Float64 = 1. )
  return 1 - (r/l)^2/( (r/l)^2 + c )
end

function cauchy( r::Float64, l::Float64, alpha::Float64, beta::Float64 )
  return ( 1 + (r/l)^alpha )^(-beta/alpha)
end
