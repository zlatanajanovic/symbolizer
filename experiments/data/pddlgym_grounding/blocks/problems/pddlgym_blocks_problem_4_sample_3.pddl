
(define (problem blocks_operator_actionsZero10) (:domain blocks_operator_actions)
  (:objects
        blue - block
	green - block
	red - block
	yellow - block
  )
(:init
	(clear blue)
	(clear red)
	(handfull)
	(holding green)
	(on red yellow)
	(ontable blue)
	(ontable yellow)
)
(:goal (and
	(on blue green)
	(on green red)
	(on red yellow)))
)
  
