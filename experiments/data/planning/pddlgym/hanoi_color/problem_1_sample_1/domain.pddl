(define (domain hanoi)
  (:requirements :strips :typing)
  
  ;; Declare the 'default' type
  (:types disc_or_peg)

  (:predicates
    (clear ?x - disc_or_peg)
    (on ?x - disc_or_peg ?y - disc_or_peg)
    (smaller ?x - disc_or_peg ?y - disc_or_peg)
  )

  (:action move
    :parameters (?disc - disc_or_peg ?from - disc_or_peg ?to - disc_or_peg)
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
