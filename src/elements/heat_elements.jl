function heat_grad_kernel()
    quote
        @into! DB = D * dNdx
        @into! GRAD_KERNEL = dNdx' * DB
        if ndim == 2 scale!(GRAD_KERNEL, t) end
    end
end

# A RHS kernel should be written such that it sets the variable
# SOURCE_KERNEL to the left hand
function heat_source_kernel()
    quote
        @into! SOURCE_KERNEL = N * eq
        if ndim == 2 scale!(SOURCE_KERNEL, t) end
    end
end

# A flux kernel should be written such that it sets the variable
# FLUX_KERNEL and CONJ_KERNEL
function heat_flux_kernel()
    quote
        @into! CONJ_KERNEL = dNdx * ed
        @into! FLUX_KERNEL = D * CONJ_KERNEL
        scale!(FLUX_KERNEL, -1)
    end
end

function get_heat_flux_size(ndim)
    if ndim == 2
        return 2
    else
        return 3
    end
end

function get_default_heat_vars(nnodes, ndim)
    Dict(:DB => (ndim ,nnodes))
end

H_S_1 = FElement(
    :heat_square_1,
    Square(),
    Lagrange{1, Square}(),
    get_default_heat_vars(4, 2),
    4,
    1,
    get_heat_flux_size(2),
    heat_grad_kernel,
    heat_source_kernel,
    heat_flux_kernel,
    (x) -> x, # Dummy func
    2)

H_S_2 = FElement(
    :heat_square_2,
    Square(),
    Serendipity{2, Square}(),
    get_default_heat_vars(8, 2),
    8,
    1,
    get_heat_flux_size(2),
    heat_grad_kernel,
    heat_source_kernel,
    heat_flux_kernel,
    (x) -> x, # Dummy func
    3)

H_T_1 = FElement(
    :heat_tri_1,
    Triangle(),
    Lagrange{1, Triangle}(),
    get_default_heat_vars(3, 2),
    3,
    1,
    get_heat_flux_size(2),
    heat_grad_kernel,
    heat_source_kernel,
    heat_flux_kernel,
    (x) -> x, # Dummy func
    1)

H_C_1 = FElement(
    :heat_cube_1,
    Cube(),
    Lagrange{1, Cube}(),
    get_default_heat_vars(8, 3),
    8,
    1,
    get_heat_flux_size(3),
    heat_grad_kernel,
    heat_source_kernel,
    heat_flux_kernel,
    (x) -> x, # Dummy func
    2)