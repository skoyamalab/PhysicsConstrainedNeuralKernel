# Efficient implementations of the spherical Bessel functions made to work with the Julia autodiff ecosystem.
# There is the formula to get the spherical Bessel functions from the Bessel functions. However, all of the
# spherical Bessel functions can be derived as functions of the trigonometric functions and polynomials, which is more efficient.

# The approach is simple: develop the derivatives by hand and explicitly write the rules to make training possible.
# Especial attention should be paid for the singularities. All singularities for the spherical Bessel functions of the first kind
# are removable singularities and thus they are whole functions in the entire complex plane.

# ======================= Implementation of the 0 order spherical Bessel function of the 1st kind ==================================
# ======================= and modified spherical Bessel function of the first kind for 0 order    ==================================

# The most important order for our application is the zero order. We determine the derivatives for up to the third, more than enough
# for most optimization methods. if the 4th derivative or higher becomes necessary, just use the same method to derive it.

# Zero order spherical Bessel function of the first kind.

function j0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return one(x)
    else
        return sin(x)/x
    end
end

# Derivatives computed up to order 3 with manually implemented derivatives with branching.
# all of these functions are continuous and infinitely differentiable, but telling the compiler
# that can be challenging.

function dj0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return zero(x)
    else
        return (x*cos(x) - sin(x))/(x^2)
    end
end

function d2j0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return -one(x)/3
    else
        return ((2-x^2)*sin(x) -2*x*cos(x))/(x^3)
    end
end

function d3j0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return zero(x)
    else
        return (3*(x^2-2)*sin(x) -x*(x^2-6)*cos(x))/(x^4)
    end
end

# We inform ChainRules.jl that the derivatives of j0, dj0 and d2j0 are
# dj0, d2j0 and d3j0 respectively using the @scalar_rule macro. Since these
# are all exact derivatives there is no need to define separate forward and backwards rules.

@scalar_rule(d2j0(x), (d3j0(x)))
@scalar_rule(dj0(x), (d2j0(x)))
@scalar_rule(j0(x), (dj0(x)))

# Dual implementations for the Bessel functions for compatibility with ForwardDiff.jl, since that is the one differentiation backend that does not work directly with ChainRules.jl

function j0(xdx::Dual{T}) where {T}
    x = value(xdx)
    dx = partials(xdx)
    return Dual{T}(j0(x), dj0(x)*dx)
end

function dj0(xdx::Dual{T}) where {T}
    x = value(xdx)
    dx = partials(xdx)
    return Dual{T}(dj0(x), d2j0(x)*dx)
end

function d2j0(xdx::Dual{T}) where {T}
    x = value(xdx)
    dx = partials(xdx)
    return Dual{T}(d2j0(x), d3j0(x)*dx)
end

# Complex dual implementation

function j0(zdz::Complex{D}) where {D<:Dual}
    z = value(zdz.re) + im*value(zdz.im)
    dz_re = partials(zdz.re)
    dz_im = partials(zdz.im)
    j = j0(z)
    dj = dj0(z)
    return D(j.re, dj.re*dz_re - dj.im*dz_im) + im*D(j.im, dj.re*dz_im + dj.im*dz_re)
end

function dj0(zdz::Complex{D}) where {D<:Dual}
    z = value(zdz.re) + im*value(zdz.im)
    dz_re = partials(zdz.re)
    dz_im = partials(zdz.im)
    j = dj0(z)
    dj = d2j0(z)
    return D(j.re, dj.re*dz_re - dj.im*dz_im) + im*D(j.im, dj.re*dz_im + dj.im*dz_re)
end

function d2j0(zdz::Complex{D}) where {D<:Dual}
    z = value(zdz.re) + im*value(zdz.im)
    dz_re = partials(zdz.re)
    dz_im = partials(zdz.im)
    j = d2j0(z)
    dj = d3j0(z)
    return D(j.re, dj.re*dz_re - dj.im*dz_im) + im*D(j.im, dj.re*dz_im + dj.im*dz_re)
end
    
# For some of our methods, the modified spherical Bessel functions are also necessary and the same principles apply.

function i0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return one(x)
    else
        return sinh(x)/x
    end
end

function di0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return zero(x)
    else
        return (x*cosh(x) - sinh(x))/(x^2)
    end
end

function d2i0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return one(x)/3
    else
        return ((x^2 + 2)*sinh(x) -2*x*cosh(x))/(x^3)
    end
end

function d3i0(x::T)::float(T) where {T <: Union{AbstractFloat, Integer, Complex{<:AbstractFloat}, Complex{<:Integer}}}
    if iszero(x)
        return zero(x)
    else
        return (x*(x^2+6)*cosh(x) - 3*(x^2+2)*sinh(x))/(x^4)
    end
end

@scalar_rule(d2i0(x), (d3i0(x)))
@scalar_rule(di0(x), (d2i0(x)))
@scalar_rule(i0(x), (di0(x)))

# Dual implementations for the modified spherical Bessel functions for compatibility with ForwardDiff.jl.

function i0(xdx::Dual{T}) where {T}
    x = value(xdx)
    dx = partials(xdx)
    return Dual{T}(i0(x), di0(x)*dx)
end

function di0(xdx::Dual{T}) where {T}
    x = value(xdx)
    dx = partials(xdx)
    return Dual{T}(di0(x), d2i0(x)*dx)
end

function d2i0(xdx::Dual{T}) where {T}
    x = value(xdx)
    dx = partials(xdx)
    return Dual{T}(d2i0(x), d3i0(x)*dx)
end

# Complex dual implementation

function i0(zdz::Complex{D}) where {D<:Dual}
    z = value(zdz.re) + im*value(zdz.im)
    dz_re = partials(zdz.re)
    dz_im = partials(zdz.im)
    j = i0(z)
    dj = di0(z)
    return D(j.re, dj.re*dz_re - dj.im*dz_im) + im*D(j.im, dj.re*dz_im + dj.im*dz_re)
end

function di0(zdz::Complex{D}) where {D<:Dual}
    z = value(zdz.re) + im*value(zdz.im)
    dz_re = partials(zdz.re)
    dz_im = partials(zdz.im)
    j = di0(z)
    dj = d2i0(z)
    return D(j.re, dj.re*dz_re - dj.im*dz_im) + im*D(j.im, dj.re*dz_im + dj.im*dz_re)
end

function d2i0(zdz::Complex{D}) where {D<:Dual}
    z = value(zdz.re) + im*value(zdz.im)
    dz_re = partials(zdz.re)
    dz_im = partials(zdz.im)
    j = d2i0(z)
    dj = d3i0(z)
    return D(j.re, dj.re*dz_re - dj.im*dz_im) + im*D(j.im, dj.re*dz_im + dj.im*dz_re)
end