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

function j0(x::T)::T where {T<:Number}
    if iszero(x)
        return one(T)
    else
        return sin(x)/x
    end
end

# Derivatives computed up to order 3 with manually implemented derivatives with branching.
# all of these functions are continuous and infinitely differentiable, but telling the compiler
# that can be challenging.

function dj0(x::T)::T where {T<:Number}
    if iszero(x)
        return zero(T)
    else
        return (x*cos(x) - sin(x))/(x^2)
    end
end

function d2j0(x::T)::T where {T<:Number}
    if iszero(x)
        return -one(T)/3
    else
        return ((2-x^2)*sin(x) -2*x*cos(x))/(x^3)
    end
end

function d3j0(x::T)::T where {T<:Number}
    if iszero(x)
        return zero(T)
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

# For some of our methods, the modified spherical Bessel functions are also necessary and the same principles apply.

function i0(x::T)::T where {T<:Number}
    if iszero(x)
        return one(T)
    else
        return sinh(x)/x
    end
end

function di0(x::T)::T where {T<: Number}
    if iszero(x)
        return zero(T)
    else
        return (x*cosh(x) - sinh(x))/(x^2)
    end
end

function d2i0(x::T)::T where {T<: Number}
    if iszero(x)
        return one(T)/3
    else
        return ((x^2 + 2)*sinh(x) -2*x*cosh(x))/(x^3)
    end
end

function d3i0(x::T)::T where {T<:Number}
    if iszero(x)
        return zero(T)
    else
        return (x*(x^2+6)*cosh(x) - 3*(x^2+2)*sinh(x))/(x^4)
    end
end

@scalar_rule(d2i0(x), (d3i0(x)))
@scalar_rule(di0(x), (d2i0(x)))
@scalar_rule(i0(x), (di0(x)))