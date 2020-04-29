#############################################################################
#Implementation of a mutable maximal Heap
#############################################################################
#Struct representing a node of the heap
struct Node{Tval, Tid, Trank}
  val::Tval
  id::Tid
  #larger ranks get picked last
  rank::Trank
end

#Mutable Heap (maximal value first)
struct MutHeap{Tval, Tid, Trank}
  # Vector containing the nodes of the heap
  nodes::Vector{Node{Tval, Tid, Trank}}
  # Vector providing a lookup for the nodes
  lookup::Vector{Tid}
end

#A function to swap two heapNodes in 
function _swap!(h,
                a,
                b)
  #Assining the new values to the lookup table 
  h.lookup[ h.nodes[a].id ] = b
  h.lookup[ h.nodes[b].id ] = a
  tempNode = h.nodes[a]
  h.nodes[a] = h.nodes[b]
  h.nodes[b] = tempNode
end

#Node comparisons
import Base.isless
function isless( a::Node, b::Node)
  isless((-a.rank, a.val), (-b.rank, b.val))
end


import Base.>=
function >=(a::Node, b::Node)
  (-a.rank, a.val) >= (-b.rank, b.val)
end

# presently not used
function >=(a::Node, b) 
  a.val >= b
end


import Base.>
function >(a::Node, b::Node) 
  (-a.rank, a.val) > (-b.rank, b.val)
end

# function >( a::Node{Tv,Ti}, b::Tv ) where {Tv,Ti}
#   a.val > b
# end


#Function that looks at element h.nodes[hInd] and moves it down the tree 
#if it is sufficiently small. Returns the new index if a move took place, 
#and lastindex(h.nodes), else
function _moveDown!(h::MutHeap, hInd)
  pivot = h.nodes[hInd]
  #If both children exist:
  if 2 * hInd + 1 <= lastindex( h.nodes )
    #If the left child is larger:
    if h.nodes[2 * hInd] >= h.nodes[ 2 * hInd + 1]
      #Check if the child is larger than the parent:
      if h.nodes[2 * hInd] >= pivot
        _swap!( h, hInd, 2 * hInd )
        return 2 * hInd
      else
        #No swap occuring:
        return lastindex( h.nodes )
      end
    #If the left child is larger:
    else
      #Check if the Child is larger than the paren:
      if h.nodes[2 * hInd + 1] >= pivot
        _swap!( h, hInd, 2 * hInd + 1 )
        return  2 * hInd + 1
      else
        #No swap occuring:
        return lastindex( h.nodes )
      end
    end
    #If only one child exists:
  elseif 2 * hInd <= lastindex( h.nodes )
    if h.nodes[2 * hInd] > pivot
      _swap!( h, hInd, 2 * hInd )
      return 2 * hInd 
    end
  end
  #No swap occuring:
  return lastindex( h.nodes )
end


#Get the leading node
function topNode(h::MutHeap)
  return first(h.nodes)
end

#Gets the leading node and sets its rank to maximum.
function topNode!( h::MutHeap{Tval,Tid,Trank} ) where {Tval, Tid, Trank}
  h.nodes[1] = Node{Tval,Tid,Trank}(h.nodes[1].val, 
                                      h.nodes[1].id,
                                      typemax(Trank))
  return first(h.nodes)
end


#Updates (decreases) an element of the heap and restores the heap property
function update!( h::MutHeap{Tv,Ti,Trank}, id::Ti, val::Tv ) where {Tv,Ti,Trank}
  tempInd::Ti = h.lookup[id]
  if h.nodes[tempInd].val > val
    h.nodes[tempInd] = typeof(h.nodes[tempInd])(val,id,h.nodes[tempInd].rank)
    while ( tempInd < lastindex( h.nodes ) )
      tempInd = _moveDown!( h, tempInd )
    end
    return val
  else
    return h.nodes[id].val
  end
end