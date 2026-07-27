(define (domain hanoi_with_types)
    (:requirements :strips :typing )
    (:types
        element ; type for disks and pegs
    )
  (:predicates (clear ?x - element) (on ?x - element ?y - element) (smaller ?x - element ?y - element))

  (:action move
    :parameters (?disc - element ?from - element ?to - element)
    :precondition (and (smaller ?to ?disc) (on ?disc ?from) 
		       (clear ?disc) (clear ?to))
    :effect  (and (clear ?from) (on ?disc ?to) (not (on ?disc ?from))  
		  (not (clear ?to))))
  )