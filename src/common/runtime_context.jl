module RuntimeContext

export set_context!, get_context, get_value, clear_context!, has_context

const _CONTEXTS = Dict{Symbol, Dict{Symbol, Any}}()

"""
    set_context!(name::Symbol; kwargs...)

Create or update a named runtime context.
"""
function set_context!(name::Symbol; kwargs...)
    context = get!(_CONTEXTS, name, Dict{Symbol, Any}())
    for (key, value) in kwargs
        context[key] = value
    end
    return context
end

"""
    has_context(name::Symbol)::Bool

Return whether the named runtime context exists.
"""
function has_context(name::Symbol)::Bool
    return haskey(_CONTEXTS, name)
end

"""
    get_context(name::Symbol)::Dict{Symbol, Any}

Fetch a named runtime context or throw if it has not been initialized.
"""
function get_context(name::Symbol)::Dict{Symbol, Any}
    haskey(_CONTEXTS, name) || error("Runtime context `$(name)` has not been initialized.")
    return _CONTEXTS[name]
end

"""
    get_value(name::Symbol, key::Symbol)

Fetch one value from a named runtime context.
"""
function get_value(name::Symbol, key::Symbol)
    context = get_context(name)
    haskey(context, key) || error("Runtime context `$(name)` does not contain key `$(key)`.")
    return context[key]
end

"""
    clear_context!(name::Symbol)

Clear one named runtime context if it exists.
"""
function clear_context!(name::Symbol)
    if haskey(_CONTEXTS, name)
        delete!(_CONTEXTS, name)
    end
    return nothing
end

end # module RuntimeContext
