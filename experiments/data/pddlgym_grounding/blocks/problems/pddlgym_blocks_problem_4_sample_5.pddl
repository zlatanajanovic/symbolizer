
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
	(handempty)
	(on blue red)
	(on red yellow)
	(ontable green)
	(ontable yellow)
)
(:goal (and
	(on blue green)
	(on green red)
	(on red yellow)))
)
  
