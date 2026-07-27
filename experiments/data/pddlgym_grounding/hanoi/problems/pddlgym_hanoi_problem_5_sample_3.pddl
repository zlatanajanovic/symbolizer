
(define (problem hanoiZero9) (:domain hanoi)
  (:objects
        d1 - default
	d2 - default
	d3 - default
	peg1 - default
	peg2 - default
	peg3 - default
  )
(:init
	(clear d1)
	(clear d3)
	(clear peg1)
	(on d1 d2)
	(on d2 peg2)
	(on d3 peg3)
	(smaller d2 d1)
	(smaller d3 d1)
	(smaller d3 d2)
	(smaller peg1 d1)
	(smaller peg1 d2)
	(smaller peg1 d3)
	(smaller peg2 d1)
	(smaller peg2 d2)
	(smaller peg2 d3)
	(smaller peg3 d1)
	(smaller peg3 d2)
	(smaller peg3 d3)
)
(:goal (and
	(on d3 peg3)
	(on d2 d3)
	(on d1 d2)))
)
  
