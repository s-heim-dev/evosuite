/*
 * Copyright (C) 2010-2018 Gordon Fraser, Andrea Arcuri and EvoSuite
 * contributors
 *
 * This file is part of EvoSuite.
 *
 * EvoSuite is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * EvoSuite is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with EvoSuite. If not, see <http://www.gnu.org/licenses/>.
 */
package org.evosuite.ga.operators.crossover;

import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.Randomness;

/**
 * Cross over with 3 parents
 *
 * @author Jonathan Berger
 */
public abstract class ThreeParentsCrossOver<T extends Chromosome<T>> extends CrossOverFunction<T> {

    /**
     * {@inheritDoc}
     * <p>
     * 
     * We split each parent in 3 thirds.
     * 
     * 
     * @param parent1
     * @param parent2
     * @param parent3
     */
    @Override
    public void crossOver(T parent1, T parent2, T parent3)
            throws ConstructionFailedException {

        if (parent1.size() < 3 || parent2.size() < 3 || parent3.size() < 3) {
            return;
        }

        T t1 = parent1.clone();
        T t2 = parent2.clone();
        T t3 = parent3.clone();
        // Choose a position in the middle
        //float splitPoint = Randomness.nextFloat();

        int oneThird1 = (int) Math.round(parent1.size() / 3.0);
        int twoThird1 = (int) Math.round((parent1.size() / 3.0)*2);

        int oneThird2 = (int) Math.round(parent2.size() / 3.0);
        int twoThird2 = (int) Math.round((parent2.size() / 3.0)*2);

        int oneThird3 = (int) Math.round(parent3.size() / 3.0);
        int twoThird3 = (int) Math.round((parent3.size() / 3.0)*2);


        parent1.crossOver(t2, t3, oneThird1, oneThird2, twoThird2, twoThird3);
        parent2.crossOver(t3, t1, oneThird2, oneThird3, twoThird3, twoThird1);
        parent3.crossOver(t1, t2, oneThird3, oneThird1, twoThird1, twoThird2);

    }
}
