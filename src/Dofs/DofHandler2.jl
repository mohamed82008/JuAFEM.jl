
# struct Field
#     name::Symbol
#     interpolation::Interpolation
#     dim::Int
# end

mutable struct DofHandler{dim,N,T,M}
    field_names::Vector{Symbol}
    field_dims::Vector{Int}
    field_interpolations::Vector{Interpolation}
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    grid::Grid{dim,N,T,M}
end

function DofHandler(grid::Grid)
    DofHandler(Symbol[], Int[], Interpolation[], Int[], Int[], ScalarWrapper(false), grid)
end

function Base.show(io::IO, dh::DofHandler)
    println(io, "DofHandler")
    println(io, "  Fields:")
    for i in 1:nfields(dh)
        println(io, "    ", repr(dh.field_names[i]), " interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
    end
    if !isclosed(dh)
        print(io, "  Not closed!")
    else
        println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
        print(io, "  Total dofs: ", ndofs(dh))
    end
end

ndofs(dh::DofHandler) = maximum(dh.cell_dofs) # TODO: This is not very nice
ndofs_per_cell(dh::DofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell] # TODO: This is not very nice
isclosed(dh::DofHandler) = dh.closed[]
nfields(dh::DofHandler) = length(dh.field_names)

function Base.push!(dh::DofHandler, name::Symbol, ip::Interpolation, dim::Int)
    @assert !isclosed(dh)
    @assert !in(name, dh.field_names)
    push!(dh.field_names, name)
    push!(dh.field_interpolations, ip)
    push!(dh.field_dims, dim)
    nothing
end

# sort and return true (was already sorted) or false (if we had to sort)
function sortedge(edge::Tuple{Int,Int})
    a, b = edge
    a < b ? (return (edge, true)) : (return ((b, a), false))
end

sortface(face::Tuple{Int,Int}) = minmax(face[1], face[2])
function sortface(face::Tuple{Int,Int,Int})
    a, b, c = face
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c)
end

# close the DofHandler and distribute all the dofs
function close!(dh::DofHandler{dim}) where {dim}
    @assert !isclosed(dh)

    # `vertexdict` keeps track of the visited vertices. We store the global vertex
    # number and the first dof we added to that vertex.
    vertexdict = ((Dict{Int,Int}() for _ in 1:nfields(dh))...)

    # `edgedict` keeps track of the visited edges, this will only be used for a 3D problem
    # An edge is determined from two vertices, but we also need to store the direction
    # of the first edge we encounter and add dofs too. When we encounter the same edge
    # the next time we check if the direction is the same, otherwise we reuse the dofs
    # in the reverse order
    edgedict = ((Dict{Tuple{Int,Int},Tuple{Int,Bool}}() for _ in 1:nfields(dh))...)

    # `facedict` keeps track of the visited faces. We only need to store the first dof we
    # added to the face; if we encounter the same face again we *always* reverse the order
    # In 2D a face (i.e. a line) is uniquely determined by 2 vertices, and in 3D a
    # face (i.e. a surface) is uniquely determined by 3 vertices.
    facedict = ((Dict{NTuple{dim,Int},Int}() for _ in 1:nfields(dh))...)

    # celldofs are never shared between different cells so there is no need
    # for a `celldict` to keep track of which cells we have added dofs too.

    nextdof = 1 # next free dof to distribute
    push!(dh.cell_dofs_offset, 1) # dofs for the first cell start at 1

    # loop over all the cells in the grid
    for (ci, cell) in enumerate(getcells(dh.grid))
        @debug println("cell #$ci")
        for (fi, interpolation) in enumerate(dh.field_interpolations)
            @debug println("  field: $(dh.field_names[fi])")
            if hasvertexdofs(interpolation)
                for vertex in vertices(cell)
                    @debug println("    vertex#$vertex")
                    if haskey(vertexdict[fi], vertex)
                        for d in 1:dh.field_dims[fi]
                            reuse_dof = vertexdict[fi][vertex] + (d-1)
                            @debug println("      reusing dof #$(reuse_dof)")
                            push!(dh.cell_dofs, reuse_dof)
                        end
                    else # add new dofs
                        for vdof in 1:nvertexdofs(interpolation) # TODO: OK to assume only 1 dof per vertex?
                            vertexdict[fi][vertex] = nextdof
                            for d in 1:dh.field_dims[fi]
                                @debug println("      adding dof#$nextdof")
                                push!(dh.cell_dofs, nextdof)
                                nextdof += 1
                            end
                        end
                    end
                end # vertex loop
            end
            if dim == 3 # edges only in 3D
                if hasedgedofs(interpolation)
                    for edge in edges(cell)
                        sedge, dir = sortedge(edge)
                        @debug println("    edge#$sedge dir: $(dir)")
                        if haskey(edgedict[fi], sedge) # reuse
                            startdof, olddir = edgedict[fi][sedge] # first dof for this edge (if dir == true)
                            for edgedof in ifelse(dir == olddir, 1:nedgedofs(interpolation), nedgedofs(interpolation):-1:1)
                                for d in 1:dh.field_dims[fi]
                                    reuse_dof = startdof + (d-1) + (edgedof-1)*dh.field_dims[fi]
                                    @debug println("      reusing dof#$(reuse_dof)")
                                    push!(dh.cell_dofs, reuse_dof)
                                end
                            end
                        else # distribute new
                            edgedict[fi][sedge] = (nextdof, dir) # store only the first dof for the edge
                            for edgedof in 1:nedgedofs(interpolation)
                                for d in 1:dh.field_dims[fi]
                                    @debug println("      adding dof#$nextdof")
                                    push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end # edge loop
                end
            end
            if hasfacedofs(interpolation)
                for face in faces(cell)
                    sface = sortface(face) # TODO: faces(cell) may as well just return the sorted list
                    @debug println("    face#$sface")
                    if haskey(facedict[fi], sface)
                        startdof = facedict[fi][sface]
                        for facedof in nfacedofs(interpolation):-1:1 # always reverse
                            for d in 1:dh.field_dims[fi]
                                reuse_dof = startdof + (d-1) + (facedof-1)*dh.field_dims[fi]
                                @debug println("      reusing dof#$(reuse_dof)")
                                push!(dh.cell_dofs, reuse_dof)
                            end
                        end
                    else # distribute new dofs
                        facedict[fi][sface] = nextdof # store the first dof for this face
                        for facedof in 1:nfacedofs(interpolation)
                            for d in 1:dh.field_dims[fi]
                                @debug println("      adding dof#$nextdof")
                                push!(dh.cell_dofs, nextdof)
                                nextdof += 1
                            end
                        end
                    end
                end
            end
            if hascelldofs(interpolation)
                @debug println("    cell#$ci")
                for celldof in 1:ncelldofs(interpolation)
                    for d in 1:dh.field_dims[fi]
                        @debug println("      adding dof#$nextdof")
                        push!(dh.cell_dofs, nextdof)
                        nextdof += 1
                    end
                end
            end
        end # field loop
        # push! the first index of the next cell to the offset vector
        push!(dh.cell_dofs_offset, length(dh.cell_dofs)+1)
    end # cell loop
    dh.closed[] = true
    return dh
end

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copy!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.grid)
    n = ndofs_per_cell(dh)
    N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; sizehint!(I, N)
    J = Int[]; sizehint!(J, N)
    global_dofs = zeros(Int, n)
    for element_id in 1:ncells
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            push!(I, dofi)
            push!(J, dofj)
        end
    end
    for d in 1:ndofs(dh)
        push!(I, d)
        push!(J, d)
    end
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end
