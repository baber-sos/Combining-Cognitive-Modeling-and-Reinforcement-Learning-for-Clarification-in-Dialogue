import itertools
import functools
import heapq

class AStarQ(object) :
    """Priority Queue that distinguishes states and nodes,
       keeps track of priority as cost plus heuristic,
       enables changing the node and priority associated with a state,
       and keeps a record of explored states."""
    
    # state label for zombie nodes in the queue
    REMOVED = '<removed-state>'     

    # sequence count lets us keep heap order stable
    # despite changes to mutable objects 
    counter = itertools.count()     
    
    @functools.total_ordering
    class QEntry(object) :
        """AStarQ.QEntry objects package together
        state, node, cost, heuristic and priority
        and enable the queue comparison operations"""

        def __init__(self, state, node, cost, heuristic) :
            self.state = state
            self.node = node
            self.cost = cost
            self.heuristic = heuristic
            if type(cost) == tuple:
                self.priority = (cost[0], cost[1] + heuristic)
            else:
                self.priority = cost + heuristic
            self.sequence = next(AStarQ.counter)
        
        def __le__(self, other) :
            return ((self.priority, self.sequence) <= 
                    (other.priority, other.sequence))
        
        def __eq__(self, other) :
            return ((self.priority, self.sequence) == 
                    (other.priority, other.sequence))
   
    def __init__(self) :
        """Set up a new problem with empty queue and nothing explored"""
        self.pq = []
        self.state_info = {} 
        self.added = 0
        self.pushed = 0

    def add_node(self, state, node, cost, heuristic):
        """Add a new state or update the priority of an existing state
           Returns outcome (added or not) for visualization"""        
        self.added = self.added + 1
        if state in self.state_info:
            already = self.state_info[state].priority
            if type(cost) == tuple:
                if already <= (cost[0], cost[1] + heuristic):
                    return False
            else:
                if already <= cost + heuristic:
                    return False
            self.remove_state_entry(state)
        entry = AStarQ.QEntry(state, node, cost, heuristic)
        self.state_info[state] = entry
        heapq.heappush(self.pq, entry)
        # print('trying to add:', self.pq)
        self.pushed = self.pushed + 1
        return True

    def remove_state_entry(self, state):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.state_info.pop(state)
        entry.state = AStarQ.REMOVED

    def pop_node(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            entry = heapq.heappop(self.pq)
            if entry.state is not AStarQ.REMOVED:
                return entry
        raise KeyError('pop from an empty priority queue')
    
    def __len__(self):
        count = 0
        for entry in self.pq:
            if entry.state is not AStarQ.REMOVED:
                count += 1
        return count
        
    def statistics(self):
        return {'added': self.added, 'pushed': self.pushed}