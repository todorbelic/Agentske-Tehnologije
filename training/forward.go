package training

import "gonum.org/v1/gonum/mat"

func (n *MLP) Forward(x mat.Matrix) (as, zs []mat.Matrix) {

	as = append(as, x)
	_x := x

	for i := 0; i < len(n.Weights); i++ {

		w := n.Weights[i]
		b := n.Biases[i]

		// z = w.x + b

		m := new(mat.Dense)

		m.Mul(_x, w)

		z := new(mat.Dense)
		addB := func(_, col int, v float64) float64 { return v + b.At(col, 0) }
		z.Apply(addB, m)

		zs = append(zs, z)

		// a = sigmoid(z)
		a := new(mat.Dense)
		a.Apply(ApplySigmoid, z)
		as = append(as, a)

		_x = a
	}

	return

}
