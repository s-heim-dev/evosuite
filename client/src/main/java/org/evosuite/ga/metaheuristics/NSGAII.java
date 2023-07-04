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
package org.evosuite.ga.metaheuristics;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.FitnessFunction;
import org.evosuite.ga.comparators.RankAndCrowdingDistanceComparator;
import org.evosuite.ga.operators.ranking.CrowdingDistance;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * NSGA-II implementation
 *
 * @author José Campos
 * @article{Deb:2002, author = {Deb, K. and Pratap, A. and Agarwal, S. and Meyarivan, T.},
 * title = {{A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II}},
 * journal = {Trans. Evol. Comp},
 * issue_date = {April 2002},
 * volume = {6},
 * number = {2},
 * month = apr,
 * year = {2002},
 * issn = {1089-778X},
 * pages = {182--197},
 * numpages = {16},
 * url = {http://dx.doi.org/10.1109/4235.996017},
 * doi = {10.1109/4235.996017},
 * acmid = {2221582},
 * publisher = {IEEE Press},
 * address = {Piscataway, NJ, USA}}
 */
public class NSGAII<T extends Chromosome<T>> extends GeneticAlgorithm<T> {
    private static final long serialVersionUID = 146182080947267628L;

    private static final Logger logger = LoggerFactory.getLogger(NSGAII.class);

    private final CrowdingDistance<T> crowdingDistance;

    /**
     * Constructor
     *
     * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object
     */
    public NSGAII(ChromosomeFactory<T> factory) {
        super(factory);
        this.crowdingDistance = new CrowdingDistance<>();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void evolve(int parentsNumber) {
        // Create the offSpring population
        List<T> offspringPopulation = new ArrayList<>(population.size());

        // execute binary tournment selection, crossover, and mutation to
        // create a offspring population Qt of size N
        for (int i = 0; i < (population.size() / 2); i++) {
            List<T> parents = new ArrayList<>();
            List<T> offsprings = new ArrayList<>();

            for (int j = 0; j < parentsNumber; j++) {
                T parent = selectionFunction.select(population);
                parents.add(parent);
                offsprings.add(parent.clone());
            }

            try {
                if (Randomness.nextDouble() <= Properties.CROSSOVER_RATE)
                    crossoverFunction.crossOver(offsprings);
            } catch (Exception e) {
                logger.info("CrossOver failed");
            }

            // Mutation
            if (Randomness.nextDouble() <= Properties.MUTATION_RATE) {
                for (T offspring : offsprings) {
                    notifyMutation(offspring);
                    offspring.mutate();
                }
            }

            // Evaluate
            for (final FitnessFunction<T> ff : this.getFitnessFunctions()) {
                for (T offspring : offsprings) {
                    ff.getFitness(offspring);
                    notifyEvaluation(offspring);
                }
            }

            for (T offspring : offsprings) {
                offspringPopulation.add(offspring);
            }
        }

        // Create the population union of Population and offSpring
        List<T> union = union(population, offspringPopulation);

        // Ranking the union
        this.rankingFunction.computeRankingAssignment(union, new LinkedHashSet<FitnessFunction<T>>(this.getFitnessFunctions()));

        int remain = population.size();
        int index = 0;
        List<T> front = null;
        population.clear();

        // Obtain the next front
        front = this.rankingFunction.getSubfront(index);

        while ((remain > 0) && (remain >= front.size())) {
            // Assign crowding distance to individuals
            this.crowdingDistance.crowdingDistanceAssignment(front, this.getFitnessFunctions());
            // Add the individuals of this front
            population.addAll(front);

            // Decrement remain
            remain = remain - front.size();

            // Obtain the next front
            index++;
            if (remain > 0)
                front = this.rankingFunction.getSubfront(index);
        }

        // Remain is less than front(index).size, insert only the best one
        if (remain > 0) {
            // front contains individuals to insert
            this.crowdingDistance.crowdingDistanceAssignment(front, this.getFitnessFunctions());

            front.sort(new RankAndCrowdingDistanceComparator<>(true));

            for (int k = 0; k < remain; k++)
                population.add(front.get(k));

            remain = 0;
        }
        //archive // TODO does it make any sense to use an archive with NSGA-II?
        /*updateFitnessFunctionsAndValues();
		for (T t : population) {
			if(t.isToBeUpdated()){
			    for (FitnessFunction<T> fitnessFunction : fitnessFunctions) {
					fitnessFunction.getFitness(t);
				}
			    t.isToBeUpdated(false);
			}
		}*/
        //
        currentIteration++;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initializePopulation() {
        logger.info("executing initializePopulation function");

        notifySearchStarted();
        currentIteration = 0;

        // Create a random parent population P0
        this.generateInitialPopulation(Properties.POPULATION);

        this.notifyIteration();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void generateSolution() {
        logger.info("executing generateSolution function");

        if (population.isEmpty())
            initializePopulation();

        while (!isFinished()) {
            evolve();
            this.notifyIteration();
            this.writeIndividuals(this.population);
        }

        notifySearchFinished();
    }

    protected List<T> union(List<T> population, List<T> offspringPopulation) {
        int newSize = population.size() + offspringPopulation.size();
        if (newSize < Properties.POPULATION)
            newSize = Properties.POPULATION;

        // Create a new population
        List<T> union = new ArrayList<>(newSize);
        union.addAll(population);

        for (int i = population.size(); i < (population.size() + offspringPopulation.size()); i++)
            union.add(offspringPopulation.get(i - population.size()));

        return union;
    }
}
