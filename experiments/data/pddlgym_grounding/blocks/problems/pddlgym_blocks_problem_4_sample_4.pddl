
(define (problem blocks_operator_actionsZero10) (:domain blocks_operator_actions)
  (:objects
        blue - block
	green - block
	red - block
	yellow - block
  )
(:init
	(clear blue)
	(clear green)
	(clear red)
	(handempty)
	(on red yellow)
	(ontable blue)
	(ontable green)
	(ontable yellow)
)
(:goal (and
	(on blue green)
	(on green red)
	(on red yellow)))
)
  
