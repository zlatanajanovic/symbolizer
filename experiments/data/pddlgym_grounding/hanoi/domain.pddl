(define (domain hanoi)
  (:requirements :strips :typing)
  
  ;; Declare the 'default' type
  (:types default)

  (:predicates
    (clear ?x - default)
    (on ?x - default ?y - default)
    (smaller ?x - default ?y - default)
  )

  (:action move
    :parameters (?disc - default ?from - default ?to - default)
    :precondition (and 
                    (smaller ?to ?disc) 
                    (on ?disc ?from)
                    (clear ?disc) 
                    (clear ?to))
    :effect  (and 
              (clear ?from) 
              (on ?disc ?to) 
              (not (on ?disc ?from))
              (not (clear ?to)))
  )
)
