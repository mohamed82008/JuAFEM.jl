"""
An `Interpolation` is used to define shape functions to interpolate
a function between nodes.

**Constructor:**

```julia
Interpolation{dim, reference_shape, order}()
```

**Arguments:**

* `dim`: the dimension the interpolation lives in
* `shape`: a reference shape, see [`AbstractRefShape`](@ref)
* `order`: the highest order term in the polynomial

The following interpolations are implemented:

* `Lagrange{1, RefCube, 1}`
* `Lagrange{1, RefCube, 2}`
* `Lagrange{2, RefCube, 1}`
* `Lagrange{2, RefCube, 2}`
* `Lagrange{2, RefTetrahedron, 1}`
* `Lagrange{2, RefTetrahedron, 2}`
* `Lagrange{3, RefCube, 1}`
* `Serendipity{2, RefCube, 2}`
* `Lagrange{3, RefTetrahedron, 1}`
* `Lagrange{3, RefTetrahedron, 2}`

**Common methods:**

* [`getnbasefunctions`](@ref)
* [`getdim`](@ref)
* [`getrefshape`](@ref)
* [`getorder`](@ref)


**Example:**

```jldoctest
julia> ip = Lagrange{2, RefTetrahedron, 2}()
JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()

julia> getnbasefunctions(ip)
6
```
"""
abstract type Interpolation{dim, shape, order} end

"""
Returns the dimension of an `Interpolation`
"""
@inline getdim(ip::Interpolation{dim}) where {dim} = dim

"""
Returns the reference shape of an `Interpolation`
"""
@inline getrefshape(ip::Interpolation{dim, shape}) where {dim, shape} = shape

"""
Returns the polynomial order of the `Interpolation`
"""
@inline getorder(ip::Interpolation{dim, shape, order}) where {dim, shape, order} = order

"""
Computes the value of the shape functions at a point ξ for a given interpolation
"""
function value(ip::Interpolation{dim}, ξ::Vec{dim}) where {dim}
    [value(ip, i, ξ) for i in 1:getnbasefunctions(ip)]
end

"""
Computes the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative(ip::Interpolation{dim}, ξ::Vec{dim, T}) where {dim, T}
    [gradient(ξ -> value(ip, i, ξ), ξ) for i in 1:getnbasefunctions(ip)]
end

#####################
# Utility functions #
#####################

"""
Returns the number of base functions for an [`Interpolation`](@ref) or `Values` object.
"""
getnbasefunctions

############
# Lagrange #
############
struct Lagrange{dim, shape, order} <: Interpolation{dim, shape, order} end

getlowerdim(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim-1,shape,order}()
getlowerorder(::Lagrange{dim,shape,order}) where {dim,shape,order} = Lagrange{dim,shape,order-1}()

##################################
# Lagrange dim 1 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{1, RefCube, 1}) = 2

function value(ip::Lagrange{1, RefCube, 1}, i::Int, ξ::Vec{1,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        i == 1 && return (T(1) - ξ_x) * T(1)/2
        i == 2 && return (T(1) + ξ_x) * T(1)/2
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 1 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{1, RefCube, 2}) = 3

function value(ip::Lagrange{1, RefCube, 2}, i::Int, ξ::Vec{1,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        i == 1 && return ξ_x * (ξ_x - T(1)) * T(1)/2
        i == 2 && return ξ_x * (ξ_x + T(1)) * T(1)/2
        i == 3 && return T(1) - ξ_x^2
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{2, RefCube, 1}) = 4

function value(ip::Lagrange{2, RefCube, 1}, i::Int, ξ::Vec{2,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        i == 1 && return (T(1) - ξ_x) * (T(1) - ξ_y) * T(1)/4
        i == 2 && return (T(1) + ξ_x) * (T(1) - ξ_y) * T(1)/4
        i == 3 && return (T(1) + ξ_x) * (T(1) + ξ_y) * T(1)/4
        i == 4 && return (T(1) - ξ_x) * (T(1) + ξ_y) * T(1)/4
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 2 RefCube order 2 #
##################################
getnbasefunctions(::Lagrange{2, RefCube, 2}) = 9

function value(ip::Lagrange{2, RefCube, 2}, i::Int, ξ::Vec{2,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        i == 1 && return (ξ_x^2 - ξ_x) * (ξ_y^2 - ξ_y) * T(1)/4
        i == 2 && return (ξ_x^2 + ξ_x) * (ξ_y^2 - ξ_y) * T(1)/4
        i == 3 && return (ξ_x^2 + ξ_x) * (ξ_y^2 + ξ_y) * T(1)/4
        i == 4 && return (ξ_x^2 - ξ_x) * (ξ_y^2 + ξ_y) * T(1)/4
        i == 5 && return (T(1) - ξ_x^2) * (ξ_y^2 - ξ_y) * T(1)/2
        i == 6 && return (ξ_x^2 + ξ_x) * (T(1) - ξ_y^2) * T(1)/2
        i == 7 && return (T(1) - ξ_x^2) * (ξ_y^2 + ξ_y) * T(1)/2
        i == 8 && return (ξ_x^2 - ξ_x) * (T(1) - ξ_y^2) * T(1)/2
        i == 9 && return (T(1) - ξ_x^2) * (T(1) - ξ_y^2)
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 1}) = 3
getlowerdim(::Lagrange{2, RefTetrahedron, order}) where {order} = Lagrange{1, RefCube, order}()

function value(ip::Lagrange{2, RefTetrahedron, 1}, i::Int, ξ::Vec{2,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        i == 1 && return ξ_x
        i == 2 && return ξ_y
        i == 3 && return T(1) - ξ_x - ξ_y
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 2}) = 6

function value(ip::Lagrange{2, RefTetrahedron, 2}, i::Int, ξ::Vec{2,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        γ = T(1) - ξ_x - ξ_y
        i == 1 && return ξ_x * (T(2)*ξ_x - T(1))
        i == 2 && return ξ_y * (T(2)*ξ_y - 1)
        i == 3 && return γ * (T(2)*γ - T(1))
        i == 4 && return T(4)*ξ_x * ξ_y
        i == 5 && return T(4)*ξ_y * γ
        i == 6 && return T(4)*ξ_x * γ
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 3 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 1}) = 4

function value(ip::Lagrange{3, RefTetrahedron, 1}, i::Int, ξ::Vec{3,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]
        i == 1 && return T(1) - ξ_x - ξ_y - ξ_z
        i == 2 && return ξ_x
        i == 3 && return ξ_y
        i == 4 && return ξ_z
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#########################################
# Lagrange dim 3 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 2}) = 10

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf
# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/AFEM.Ch10.pdf
function value(ip::Lagrange{3, RefTetrahedron, 2}, i::Int, ξ::Vec{3,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]
        i == 1  && return (T(-2) * ξ_x - T(2) * ξ_y - T(2) * ξ_z + T(1)) * (-ξ_x - ξ_y - ξ_z + T(1))
        i == 2  && return ξ_x * (T(2) * ξ_x - T(1))
        i == 3  && return ξ_y * (T(2) * ξ_y - T(1))
        i == 4  && return ξ_z * (T(2) * ξ_z - T(1))
        i == 5  && return ξ_x * (T(-4) * ξ_x - T(4) * ξ_y - T(4) * ξ_z + T(4))
        i == 6  && return T(4) * ξ_x * ξ_y
        i == 7  && return T(4) * ξ_y * (-ξ_x - ξ_y - ξ_z + T(1))
        i == 8  && return ξ_z * (T(-4) * ξ_x - T(4) * ξ_y - T(4) * ξ_z + T(4))
        i == 9  && return T(4) * ξ_x * ξ_z
        i == 10 && return T(4) * ξ_y * ξ_z
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

##################################
# Lagrange dim 3 RefCube order 1 #
##################################
getnbasefunctions(::Lagrange{3, RefCube, 1}) = 8

function value(ip::Lagrange{3, RefCube, 1}, i::Int, ξ::Vec{3,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]
        i == 1 && return T(1)/8(T(1) - ξ_x) * (T(1) - ξ_y) * (T(1) - ξ_z)
        i == 2 && return T(1)/8(T(1) + ξ_x) * (T(1) - ξ_y) * (T(1) - ξ_z)
        i == 3 && return T(1)/8(T(1) + ξ_x) * (T(1) + ξ_y) * (T(1) - ξ_z)
        i == 4 && return T(1)/8(T(1) - ξ_x) * (T(1) + ξ_y) * (T(1) - ξ_z)
        i == 5 && return T(1)/8(T(1) - ξ_x) * (T(1) - ξ_y) * (T(1) + ξ_z)
        i == 6 && return T(1)/8(T(1) + ξ_x) * (T(1) - ξ_y) * (T(1) + ξ_z)
        i == 7 && return T(1)/8(T(1) + ξ_x) * (T(1) + ξ_y) * (T(1) + ξ_z)
        i == 8 && return T(1)/8(T(1) - ξ_x) * (T(1) + ξ_y) * (T(1) + ξ_z)
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

###############
# Serendipity #
###############
struct Serendipity{dim, shape, order} <: Interpolation{dim, shape, order} end

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{2, RefCube, 2}) = 8
getlowerdim(::Serendipity{2, RefCube, 2}) = Lagrange{1, RefCube, 2}()
getlowerorder(::Serendipity{2, RefCube, 2}) = Lagrange{2, RefCube, 1}()

function value(ip::Serendipity{2, RefCube, 2}, i::Int, ξ::Vec{2,T}) where {T}
    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        i == 1 && return (T(1) - ξ_x) * (T(1) - ξ_y) * T(1)/4(-ξ_x - ξ_y - T(1))
        i == 2 && return (T(1) + ξ_x) * (T(1) - ξ_y) * T(1)/4( ξ_x - ξ_y - T(1))
        i == 3 && return (T(1) + ξ_x) * (T(1) + ξ_y) * T(1)/4( ξ_x + ξ_y - T(1))
        i == 4 && return (T(1) - ξ_x) * (T(1) + ξ_y) * T(1)/4(-ξ_x + ξ_y - T(1))
        i == 5 && return T(1)/2*(T(1) - ξ_x * ξ_x) * (T(1) - ξ_y)
        i == 6 && return T(1)/2*(T(1) + ξ_x) * (T(1) - ξ_y * ξ_y)
        i == 7 && return T(1)/2(T(1) - ξ_x * ξ_x) * (T(1) + ξ_y)
        i == 8 && return T(1)/2(T(1) - ξ_x) * (T(1) - ξ_y * ξ_y)
    end
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
