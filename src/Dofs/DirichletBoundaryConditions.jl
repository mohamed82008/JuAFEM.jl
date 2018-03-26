"""
    DirichletBoundaryConditions

A Dirichlet boundary condition is a boundary where the solution is fixed to take a certain value.
The struct `DirichletBoundaryConditions` represents a collection of such boundary conditions.

It is created from a `DofHandler`

```jldoctest dbc
julia> dbc = DirichletBoundaryConditions(dh)
```

Dirichlet boundary conditions are added to certain components of a field for a specific nodes of the grid.
A function is also given that should be of the form `(x,t) -> v` where `x` is the coordinate of the node, `t` is a
time parameter and `v` should be of the same length as the number of components the bc is applied to:

```jldoctest
julia> addnodeset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0);

julia> nodes = grid.nodesets["clamped"]

julia> push!(dbc, :temperature, nodes, (x,t) -> t * [x[2], 0.0, 0.0], [1, 2, 3])
```

Boundary conditions are now updates by specifying a time:

```jldoctest
julia> t = 1.0;

julia> update!(dbc, t)
```

The boundary conditions can be applied to a vector:

```jldoctest
julia> u = zeros(ndofs(dh))

julia> apply!(u, dbc)
```

"""
struct DirichletBoundaryCondition{TI}
    f::Function
    nodes::Set{TI}
    field::Symbol
    components::Vector{TI}
    idxoffset::TI
end

struct DirichletBoundaryConditions{DH <: DofHandler, T, TI}
    bcs::Vector{DirichletBoundaryCondition{TI}}
    dofs::Vector{TI}
    free_dofs::Vector{TI}
    values::Vector{T}
    dofmapping::Dict{TI, TI} # global dof -> index into dofs and values
    dh::DH
    closed::ScalarWrapper{Bool}
end
function DirichletBoundaryConditions(dh::DofHandler{dim, N, T, M, TI}) where {dim, N, T, M, TI<:Integer}
    @assert isclosed(dh)
    DirichletBoundaryConditions(DirichletBoundaryCondition{TI}[], TI[], TI[], T[], Dict{TI,TI}(), dh, ScalarWrapper(false))
end

function Base.show(io::IO, dbcs::DirichletBoundaryConditions)
    println(io, "DirichletBoundaryConditions:")
    if !isclosed(dbcs)
        print(io, "  Not closed!")
    else
        println(io, "  BCs:")
        for dbc in dbcs.bcs
            println(io, "    ", "Field: ", dbc.field, " ", "Components: ", dbc.components)
        end
    end
end

isclosed(dbcs::DirichletBoundaryConditions) = dbcs.closed[]
dirichlet_dofs(dbcs::DirichletBoundaryConditions) = dbcs.dofs
free_dofs(dbcs::DirichletBoundaryConditions) = dbcs.free_dofs
function close!(dbcs::DirichletBoundaryConditions)
    fill!(dbcs.values, NaN)
    fdofs = Array(setdiff(dbcs.dh.dofs_nodes, dbcs.dofs))
    resize!(dbcs.free_dofs, length(fdofs))
    copy!(dbcs.free_dofs, fdofs)
    for i in 1:length(dbcs.dofs)
        dbcs.dofmapping[dbcs.dofs[i]] = i
    end

    dbcs.closed[] = true

    return dbcs
end

function add!(dbcs::DirichletBoundaryConditions, field::Symbol,
                          nodes::Union{Set{TI}, Vector{TI}}, f::Function, component::TI=1) where {TI<:Integer}
    add!(dbcs, field, nodes, f, [component])
end

# Adds a boundary condition
function add!(dbcs::DirichletBoundaryConditions, field::Symbol,
                          nodes::Union{Set{TI}, Vector{TI}}, f::Function, components::Vector{TI}) where {TI<:Integer}
    field in dbcs.dh.field_names || error("field $field does not exist in the dof handler, existing fields are $(dh.field_names)")
    for component in components
        0 < component <= ndim(dbcs.dh, field) || error("component $component is not within the range of field $field which has $(ndim(dbcs.dh, field)) dimensions")
    end

    if length(nodes) == 0
        warn("added Dirichlet BC to node set containing 0 nodes")
    end

    dofs_bc = TI[]
    offset = dof_offset(dbcs.dh, field)
    for node in nodes
        for component in components
            dofid = dbcs.dh.dofs_nodes[offset + component, node]
            push!(dofs_bc, dofid)
        end
    end

    n_bcdofs = length(dofs_bc)

    append!(dbcs.dofs, dofs_bc)
    idxoffset = length(dbcs.values)
    resize!(dbcs.values, length(dbcs.values) + n_bcdofs)

    push!(dbcs.bcs, DirichletBoundaryCondition{TI}(f, Set(nodes), field, components, idxoffset))

end

# Updates the DBC's to the current time `time`
function update!(dbcs::DirichletBoundaryConditions, time::Real=0.0)
    @assert dbcs.closed[]
    bc_offset = 0
    for dbc in dbcs.bcs
        # Function barrier
        _update!(dbcs.values, dbc.f, dbc.nodes, dbc.field,
                 dbc.components, dbcs.dh, dbc.idxoffset, dbcs.dofmapping, time)
    end
end

function _update!(values::Vector{T}, f::Function, nodes::Set{TI}, field::Symbol,
                  components::Vector{TI}, dh::DofHandler, idx_offset::TI,
                  dofmapping::Dict{TI,TI}, time::Real) where {T, TI<:Integer}
    mesh = dh.grid
    offset = dof_offset(dh, field)
    for node in nodes
        x = getcoordinates(getnodes(mesh, node))
        bc_value = f(x, time)
        @assert length(bc_value) == length(components)
        for i in 1:length(components)
            c = components[i]
            dof_number = dh.dofs_nodes[offset + c, node]
            dbc_index = dofmapping[dof_number]
            values[dbc_index] = bc_value[i]
        end
    end
end

# Saves the dirichlet boundary conditions to a vtkfile.
# Values will have a 1 where bcs are active and 0 otherwise
function WriteVTK.vtk_point_data(vtkfile, dbcs::DirichletBoundaryConditions)
    unique_fields = []
    for dbc in dbcs.bcs
        push!(unique_fields, dbc.field)
    end
    unique_fields = unique(unique_fields)

    for field in unique_fields
        nd = ndim(dbcs.dh, field)
        data = zeros(Float64, nd, getnnodes(dbcs.dh.grid))
        for dbc in dbcs.bcs
            if dbc.field != field
                continue
            end

            for node in dbc.nodes
                for component in dbc.components
                    data[component, node] = 1.0
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field)*"_bc")
    end
    return vtkfile
end

function apply!(v::Vector, bc::DirichletBoundaryConditions)
    @assert length(v) == ndofs(bc.dh)
    v[bc.dofs] = bc.values
    return v
end

function apply_zero!(v::Vector, bc::DirichletBoundaryConditions)
    @assert length(v) == ndofs(bc.dh)
    v[bc.dofs] = 0
    return v
end

function apply!(K::Union{SparseMatrixCSC, Symmetric}, bc::DirichletBoundaryConditions)
    apply!(K, eltype(K)[], bc, true)
end

function apply_zero!(K::Union{SparseMatrixCSC, Symmetric}, f::AbstractVector, bc::DirichletBoundaryConditions)
    apply!(K, f, bc, true)
end

function apply!(KK::Union{SparseMatrixCSC, Symmetric}, f::AbstractVector, bc::DirichletBoundaryConditions, applyzero::Bool=false)
    K = isa(KK, Symmetric) ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, bc.dofs, bc.dofs)
    @boundscheck length(f) == 0 || checkbounds(f, bc.dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver
    @inbounds for i in 1:length(bc.values)
        d = bc.dofs[i]
        v = bc.values[i]

        if !applyzero && v != 0
            for j in nzrange(K, d)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    K[:, bc.dofs] = 0
    K[bc.dofs, :] = 0
    @inbounds for i in 1:length(bc.values)
        d = bc.dofs[i]
        v = bc.values[i]
        K[d, d] = m
        # We will only enter here with an empty f vector if we have assured that v == 0 for all dofs
        if length(f) != 0
            vz = applyzero ? zero(eltype(f)) : v
            f[d] = vz * m
        end
    end
end

function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end
