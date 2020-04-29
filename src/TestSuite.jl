using Test
using ForwardDiff
using Rotations
include("./BEMUtils.jl")

function primitive(x, h, r)
  η = sqrt(h^2 + r^2 * x^2 + x^2)
  # return x * log(η / sqrt(h^2 + x^2) + r * x / sqrt(h^2 + x^2)) -
  # 1/2 * h * atan((h * (η^2 - r^3 * x * η)) / ((r^2 * (x^2 + h^2) - η^2) * η^2 - h^2 * r^4 * x^2))

  return x * log(η / sqrt(h^2 + x^2) + r * x / sqrt(h^2 + x^2)) -
  1/2 * h * angle((r^2 - 1) * h^2 - (r^2 + 1) * x^2 + im * 2 * h *r * η) 

end

function deriv(x, h, r)
  return log(sqrt(1 + r^2 * x^2 / (x^2 + h^2)) + r * x / sqrt(x^2 + h^2))
end

function grad_primitive(x, h, r) 
  return ForwardDiff.derivative(x -> primitive(x, h, r), x)
end

function ∫Φ_rectangle_numeric(a, b, h)
  Φ(xy) = 1/sqrt(xy[1]^2 + xy[2]^2 + h^2)
  cb_result = hcubature(Φ, (0.0,0.0), (a,b))
  return cb_result[1]
end

function ∫Φ_rectangle(a, b, h)
  return ∫Φ_right_angle(a, b, h) + ∫Φ_right_angle(b, a, h)
end


function ∫Φ_right_angle_test()
  a, b, h = rand(3)
  # @show ∫Φ_rectangle_numeric(a, b, h) - ∫Φ_rectangle(a, b, h) 
  return ∫Φ_rectangle_numeric(a, b, h) ≈ ∫Φ_rectangle(a, b, h) 
end

function ∫Φ_general_angle_test()
  a, b, d, h = rand(4) .- 0.5
  a⃗ = SVector(a, d)
  α⃗ = SVector(a, 0.)
  b⃗ = SVector(b, d)
  β⃗ = SVector(b, 0.)

  return abs(∫Φ_rectangle_numeric(abs(a), abs(d), h) -
             sign(a * b) * ∫Φ_rectangle_numeric(abs(b), abs(d), h)) ≈
         abs(sign(a * b) * ∫Φ_general_angle(α⃗, a⃗, h) -
             ∫Φ_general_angle(b⃗, β⃗, h)) +
         ∫Φ_general_angle(a⃗, b⃗, h)
end

function ∫Φ_square_test()
  c1, c2, h = rand(3) .-0.5; d = rand()
  c1 *= 4; c2 *= 4
  c⃗ = SVector(c1, c2)

  Φ(xy) = 1/sqrt(xy[1]^2 + xy[2]^2 + h^2)
  cb_result = hcubature(Φ, (c1 - d/2, c2 - d/2), (c1 + d/2, c2 + d/2))
  cb_result[1]
  ∫Φ_square(c⃗, d, h)
  return ∫Φ_square(c⃗, d, h) ≈ cb_result[1]
end

function matrix_element_point_scaling_test()
  d = rand()
  v_point = rand(3)
  v_scaling = rand(3)
  O = rand(RotMatrix{3})
  point = Point(v_point)
  scaling = rotate(HaarScaling(d, v_scaling), O)

  @show constructMatrixElement(point, scaling)
  @show constructMatrixElement_numeric(point, scaling)


  return constructMatrixElement(point, scaling) ≈ constructMatrixElement_numeric(point, scaling)
end

function matrix_element_scaling_scaling_test()
  d1, d2 = rand(), rand()
  v1, v2 = rand(3), rand(3)
  O1, O2 = rand(RotMatrix{3}), rand(RotMatrix{3})
  e1 = rotate(HaarScaling(d1, v1), O1)
  e2 = rotate(HaarScaling(d2, v2), O2)

  @show constructMatrixElement(e1, e2)
  @show constructMatrixElement_numeric(e1, e2)
  return constructMatrixElement(e1, e2) ≈ constructMatrixElement_numeric(e1, e2)
end

@test ∫Φ_right_angle_test()
@test ∫Φ_general_angle_test()
@test ∫Φ_square_test()
@test matrix_element_point_scaling_test()
@test matrix_element_scaling_scaling_test()