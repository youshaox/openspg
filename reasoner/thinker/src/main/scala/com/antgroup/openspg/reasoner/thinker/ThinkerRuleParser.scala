/*
 * Copyright 2023 OpenSPG Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied.
 */

package com.antgroup.openspg.reasoner.thinker

import java.util.Locale

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import com.antgroup.openspg.reasoner.KGDSLParser._
import com.antgroup.openspg.reasoner.lube.common.expr.{
  BEqual,
  BinaryOpExpr,
  BNotEqual,
  ConceptExpr,
  Expr,
  TripleExpr
}
import com.antgroup.openspg.reasoner.lube.utils.transformer.impl.Expr2QlexpressTransformer
import com.antgroup.openspg.reasoner.parser.expr.RuleExprParser
import com.antgroup.openspg.reasoner.thinker.logic.graph.{Element, Entity, Predicate}
import com.antgroup.openspg.reasoner.thinker.logic.graph
import com.antgroup.openspg.reasoner.thinker.logic.rule.{
  ClauseEntry,
  EntityPattern,
  Node,
  TriplePattern
}
import com.antgroup.openspg.reasoner.thinker.logic.rule.exact.{
  And,
  Condition,
  Not,
  Or,
  QlExpressCondition
}
import org.apache.commons.lang3.StringUtils

class ThinkerRuleParser extends RuleExprParser {
  val expr2StringTransformer = new Expr2QlexpressTransformer()
  var defaultAliasNum = 0
  val aliasToElementMap = new mutable.HashMap[String, Element]()

  val conditionToElementMap: mutable.HashMap[Condition, mutable.HashSet[ClauseEntry]] =
    new mutable.HashMap()

  def thinkerParseValueExpression(
      ctx: Value_expressionContext,
      body: ListBuffer[ClauseEntry]): Node = {
    ctx.getChild(0) match {
      case c: Logic_value_expressionContext => thinkerParseLogicValueExpression(c, body)
      case c: Project_value_expressionContext => thinkerParseProjectValueExpression(c, body)
    }
  }

  def thinkerParseLogicValueExpression(
      ctx: Logic_value_expressionContext,
      body: ListBuffer[ClauseEntry]): Node = {
    val orNodeList: List[Node] =
      ctx.logic_term().asScala.toList.map(x => thinkerParseLogicTerm(x, body))
    if (orNodeList.length > 1) {
      new Or(orNodeList.asJava)
    } else {
      orNodeList.head
    }
  }

  def thinkerParseLogicTerm(ctx: Logic_termContext, body: ListBuffer[ClauseEntry]): Node = {
    val andNodeList: List[Node] =
      ctx.logic_factor().asScala.toList.map(x => thinkerParseLogicFactor(x, body))
    if (andNodeList.length > 1) {
      new And(andNodeList.asJava)
    } else {
      andNodeList.head
    }
  }

  def thinkerParseLogicFactor(ctx: Logic_factorContext, body: ListBuffer[ClauseEntry]): Node = {
    var resultNode = thinkerParseLogicTest(ctx.logic_test(), body)
    if (ctx.not() != null) {
      resultNode = new Not(resultNode)
    }
    resultNode
  }

  def insertIntoMap(condition: Condition, element: Set[ClauseEntry]): Unit = {
    val existElementSet: mutable.Set[ClauseEntry] =
      conditionToElementMap.getOrElseUpdate(condition, mutable.HashSet())
    for (e <- element) {
      if (e.isInstanceOf[EntityPattern]) {
        existElementSet += e
      }
    }
  }

  def thinkerParseLogicTest(ctx: Logic_testContext, body: ListBuffer[ClauseEntry]): Node = {
    val newBody: ListBuffer[ClauseEntry] = new ListBuffer[ClauseEntry]()
    val resultNode: Node = ctx.getChild(0) match {
      case c: Spo_ruleContext =>
        val (subject, predicate, object_) = parseSpoRule(c)
        newBody += new TriplePattern(new logic.graph.Triple(subject, predicate, object_))
        new QlExpressCondition(expr2StringTransformer.transform(parseLogicTest(ctx)).head)
      case c: Concept_nameContext =>
        val conceptEntity = constructConceptEntity(c)
        newBody += new EntityPattern(conceptEntity)
        new QlExpressCondition(expr2StringTransformer.transform(parseLogicTest(ctx)).head)
      case c: ExprContext => thinkerParseExpr(c, newBody)
    }
    body ++= newBody
    if (resultNode.isInstanceOf[Condition]) {
      insertIntoMap(resultNode.asInstanceOf[Condition], newBody.toSet)
    }
    resultNode
  }

  def getAliasFromElement(element: Element): String = {
    element match {
      case e: Entity => e.getAlias
      case p: Predicate => p.getAlias
      case n: logic.graph.Node => n.getAlias
      case _ => throw new IllegalArgumentException("%s element has no alias".format(element))
    }
  }

  def refactorConceptName(concept: Concept_nameContext): String = {
    val metaConceptName = concept.meta_concept_type().getText
    val conceptInstanceIdName =
      concept.concept_instance_id().getText.stripPrefix("`").stripSuffix("`")
    metaConceptName + "/" + conceptInstanceIdName
  }

  override def parseLogicTest(ctx: Logic_testContext): Expr = {
    val bExpr: Expr = ctx.getChild(0) match {
      case c: Spo_ruleContext =>
        val (subject, predicate, object_) = parseSpoRule(c)
        TripleExpr(
          getAliasFromElement(subject),
          getAliasFromElement(predicate),
          getAliasFromElement(object_))
      case concept: Concept_nameContext =>
        ConceptExpr(refactorConceptName(concept))
      case expr: ExprContext => parseExpr(expr)
    }
    Option(ctx.getChild(1)) match {
      case Some(x) =>
        val tExpr: Expr = str2bool(ctx.truth_value().getText)
        val compareStr = x match {
          case _: Equals_operatorContext => BEqual
          case _: Not_equals_operatorContext => BNotEqual
          case _ =>
            ctx.getChild(2).getText.toUpperCase(Locale.ROOT) match {
              case "NOT" => BNotEqual
              case _ => BEqual
            }
        }
        BinaryOpExpr(compareStr, bExpr, tExpr)
      case None =>
        bExpr
    }
  }

  def constructConceptEntity(ctx: Concept_nameContext): Entity = {
    new Entity(
      removeGraveAccentInConceptId(ctx.concept_instance_id()),
      ctx.meta_concept_type().getText)
  }

  def thinkerParseExpr(ctx: ExprContext, body: ListBuffer[ClauseEntry]): Node = {
    ctx.getChild(0) match {
      case c: Binary_exprContext => thinkerParseBinaryExpr(c, body)
      case c: Unary_exprContext => thinkerParseUnaryExpr(c, body)
      case c: Function_exprContext => thinkerParseFunctionExpr(c, body)
    }
  }

  def thinkerParseBinaryExpr(ctx: Binary_exprContext, body: ListBuffer[ClauseEntry]): Node = {
    ctx
      .project_value_expression()
      .asScala
      .foreach(x => thinkerParseProjectValueExpression(x, body))
    new QlExpressCondition(
      expr2StringTransformer
        .transform(super.parseBinaryExpr(ctx))
        .head)
  }

  def thinkerParseProjectValueExpression(
      ctx: Project_value_expressionContext,
      body: ListBuffer[ClauseEntry]): Node = {
    ctx.term().asScala.foreach(term => thinkerParseTerm(term, body))
    new QlExpressCondition(
      expr2StringTransformer.transform(super.parseProjectValueExpression(ctx)).head)
  }

  def thinkerParseTerm(ctx: TermContext, body: ListBuffer[ClauseEntry]): Unit = {
    ctx.factor().asScala.foreach(factor => thinkerParseFactor(factor, body))
  }

  def thinkerParseFactor(ctx: FactorContext, body: ListBuffer[ClauseEntry]): Unit = {
    thinkerParseProjectPrimary(ctx.project_primary(), body)
  }

  def thinkerParseProjectPrimary(
      ctx: Project_primaryContext,
      body: ListBuffer[ClauseEntry]): Unit = {
    ctx.getChild(0) match {
      case c: Concept_nameContext =>
        body += new EntityPattern(constructConceptEntity(c))
      case c: Value_expression_primaryContext =>
        thinkerParseValueExpressionPrimary(c, body)
      case c: Numeric_value_functionContext =>
        thinkerParseNumericFunction(c, body)
    }
  }

  override def parseProjectPrimary(ctx: Project_primaryContext): Expr = {
    ctx.getChild(0) match {
      case c: Concept_nameContext =>
        ConceptExpr(refactorConceptName(c))
      case c: Value_expression_primaryContext =>
        parseValueExpressionPrimary(c)
      case c: Numeric_value_functionContext =>
        parseNumericFunction(c)
    }
  }

  def thinkerParseValueExpressionPrimary(
      ctx: Value_expression_primaryContext,
      body: ListBuffer[ClauseEntry]): Unit = {
    ctx.getChild(0) match {
      case c: Parenthesized_value_expressionContext =>
        thinkerParseValueExpression(c.value_expression(), body)
      case c: Non_parenthesized_value_expression_primary_with_propertyContext =>
        thinkerParseNonParentValueExpressionPrimaryWithProperty(c, body)
    }
  }

  def thinkerParseNonParentValueExpressionPrimaryWithProperty(
      ctx: Non_parenthesized_value_expression_primary_with_propertyContext,
      body: ListBuffer[ClauseEntry]): Unit = {
    ctx.non_parenthesized_value_expression_primary().getChild(0) match {
      case c: Function_exprContext => thinkerParseFunctionExpr(c, body)
      case c: Binding_variableContext =>
        thinkParseBindingVariable(c, body, ctx.property_name().asScala.toList)
      case _ =>
    }
  }

  def thinkParseBindingVariable(
      ctx: Binding_variableContext,
      body: ListBuffer[ClauseEntry],
      propertyNameList: List[Property_nameContext]): Unit = {
    if (propertyNameList != null && propertyNameList.nonEmpty) {
      val alias: String = ctx.binding_variable_name().getText
      if (!aliasToElementMap.contains(alias)) {
        throw new IllegalArgumentException("alias %s not define".format(alias))
      }
      if (propertyNameList.size > 1) {
        throw new IllegalArgumentException(
          "Variable %s can not have multilevel attributes".format(alias))
      }
      val propertyName: String = propertyNameList.head.getText
      val subject = aliasToElementMap(alias)
      val predicate = new Predicate(propertyName)
      val o = new graph.Any()
      body += new TriplePattern(new logic.graph.Triple(subject, predicate, o))
    }
  }

  def thinkerParseFunctionExpr(ctx: Function_exprContext, body: ListBuffer[ClauseEntry]): Node = {
    val argsContext: Function_argsContext = ctx.function_args()
    if (null != argsContext) {
      argsContext
        .list_element_list()
        .list_element()
        .asScala
        .foreach(x => {
          this.thinkerParseValueExpression(x.value_expression(), body)
        })
    }
    new QlExpressCondition(expr2StringTransformer.transform(super.parseFunctionExpr(ctx)).head)
  }

  def thinkerParseNumericFunction(
      ctx: Numeric_value_functionContext,
      body: ListBuffer[ClauseEntry]): Unit = {
    ctx.getChild(0) match {
      case c: Absolute_value_expressionContext =>
        thinkerParseProjectValueExpression(c.project_value_expression(), body)
      case c: Floor_functionContext =>
        thinkerParseProjectValueExpression(c.project_value_expression(), body)
      case c: Ceiling_functionContext =>
        thinkerParseProjectValueExpression(c.project_value_expression(), body)
    }
  }

  def thinkerParseUnaryExpr(ctx: Unary_exprContext, body: ListBuffer[ClauseEntry]): Node = {
    this.thinkerParseValueExpression(ctx.value_expression(), body)
  }

  def removeGraveAccentInConceptId(concept_instance_idCtx: Concept_instance_idContext): String = {
    concept_instance_idCtx.getText.stripPrefix("`").stripSuffix("`")
  }

  def parseSpoRule(ctx: Spo_ruleContext, isHead: Boolean = false): (Element, Element, Element) = {
    val sNode: Element = parseNode(ctx, 0, isHead)
    val pPredicate: Element = parsePredicate(ctx)
    val oNode: Element = parseNode(ctx, 1)
    (sNode, pPredicate, oNode)
  }

  def parseNode(ctx: Spo_ruleContext, index: Int, isHead: Boolean = false): Element = {
    val elementPatternDeclaration =
      ctx.node_pattern(index).element_pattern_declaration_and_filler()
    val (sAlias, sType) =
      parseElement_pattern_declaration_and_fillerContext(elementPatternDeclaration, isHead)
    if (aliasToElementMap.contains(sAlias)) {
      return aliasToElementMap(sAlias)
    }
    var sNode: Element = new graph.Node(sType, sAlias)
    if (StringUtils.isNotEmpty(sType)) {
      val conceptContext = elementPatternDeclaration
        .element_lookup()
        .label_expression()
        .label_name()
        .concept_name()
      if (conceptContext != null) {
        val conceptEntity = constructConceptEntity(conceptContext)
        conceptEntity.setAlias(sAlias)
        sNode = conceptEntity
      }
    }
    aliasToElementMap += (sAlias -> sNode)
    sNode
  }

  def parsePredicate(ctx: Spo_ruleContext): Element = {
    val (pAlias, pType) = parseElement_pattern_declaration_and_fillerContext(
      ctx.rule_name_declaration().element_pattern_declaration_and_filler())
    var pPredicate: Element = null
    if (aliasToElementMap.contains(pAlias)) {
      pPredicate = aliasToElementMap(pAlias)
    } else {
      pPredicate = new Predicate(pType, pAlias)
      aliasToElementMap += (pAlias -> pPredicate)
    }
    pPredicate
  }

  def parseElement_pattern_declaration_and_fillerContext(
      ctx: Element_pattern_declaration_and_fillerContext,
      isHead: Boolean = false): (String, String) = {
    var alias = ""
    if (ctx.element_variable_declaration() != null) {
      alias = ctx.element_variable_declaration().element_variable().getText
    } else {
      alias = "anonymous_" + getDefaultAliasNum
    }
    var labelName = ""
    if (ctx.element_lookup() != null) {
      labelName = ctx.element_lookup().label_expression().label_name().getText
    }
    if (!isHead && StringUtils.isBlank(labelName) && !aliasToElementMap.contains(alias)) {
      throw new IllegalArgumentException("alias " + alias + " is not defined")
    }
    (alias, labelName)
  }

  def getDefaultAliasNum: String = {
    defaultAliasNum += 1
    defaultAliasNum.toString
  }

}
