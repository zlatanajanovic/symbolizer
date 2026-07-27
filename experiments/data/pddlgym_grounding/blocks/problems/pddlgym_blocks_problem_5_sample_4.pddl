
(define (problem blocks_operator_actionsOne3) (:domain blocks_operator_actions)
  (:objects
        blue - block
	green - block
	grey - block
	red - block
	yellow - block
  )
(:init
	(clear blue)
	(clear green)
	(handfull)
	(holding grey)
	(on green red)
	(on red yellow)
	(ontable blue)
	(ontable yellow)
)
(:goal (and
	(on grey blue)
	(on blue green)
	(on yellow red)
	(clear yellow)
	(clear grey)))
)
  
